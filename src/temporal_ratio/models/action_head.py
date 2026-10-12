import logging
import math
import os
from collections import OrderedDict
from einops import rearrange

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import temporal_ratio.models.gemma as _gemma
from temporal_ratio.models.paligemma_with_expert import PaliGemmaWithExpertModel

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def drop_first_frame_tokens(pad_masks: Tensor, num_latent_frames: int, prob: float) -> Tensor:
    """Hide the tokens of the first latent frame, independently per sample with probability ``prob``.

    The first latent frame is the clean conditioning frame (the current observation); hiding it makes
    the action tokens use the DiT features of the predicted frames, which are still conditioned on it.

    Args:
      pad_masks: bool[B, T * S] video-token masks for T latent frames of S tokens each.
    """
    bsize, seq_len = pad_masks.shape
    drop = torch.rand(bsize, device=pad_masks.device) < prob
    first = torch.arange(seq_len, device=pad_masks.device) < seq_len // num_latent_frames
    return pad_masks & ~(drop[:, None] & first[None, :])


class AdaLn(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embeddings."""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        
        # Condition projection - projects to 2*dim for scale and shift
        self.cond_proj = nn.Linear(dim, 2 * dim)
        
    def forward(self, x, cond):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cond: Conditioning tensor of shape (batch, dim) - typically timestep embeddings
        
        Returns:
            Normalized and modulated tensor of shape (batch, seq_len, dim)
        """
        # Apply layer normalization
        x_norm = self.ln(x)
        
        # Project condition to scale and shift
        cond_projected = self.cond_proj(cond)
        cond_scale, cond_shift = cond_projected.chunk(2, dim=-1)
        
        # Expand to match x dimensions: (batch, dim) -> (batch, 1, dim)
        cond_scale = cond_scale.unsqueeze(1)
        cond_shift = cond_shift.unsqueeze(1)
        
        # Apply adaptive modulation: (1 + cond_scale) * x_norm + cond_shift
        return (1 + cond_scale) * x_norm + cond_shift


class VideoVLAPytorch(nn.Module):
    def __init__(self, config, video_latent_shape=None, video_feature_dim=None):
        super().__init__()
        self.config = config
        self.action_horizon = config.action_horizon
        self.interleaved = config.interleaved
        self.multimodal = config.multimodal
        self.video_latent_shape = video_latent_shape

        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.action_expert_width = action_expert_config.width
        self.video_expert_width = 2*action_expert_config.width

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            action_expert_config,
            use_adarms=True,
            precision=config.dtype,
            multimodal=self.multimodal
        )

        # 2x2 patchify multiplies feature dim by 4
        embedder_in_dim = 2 * 2 * (video_feature_dim if video_feature_dim is not None else self.video_expert_width)
        self.state_proj = nn.Linear(getattr(config, 'state_dim', 32), action_expert_config.width)
        self.video_latent_embedder = nn.Linear(embedder_in_dim, action_expert_config.width)
        self.video_adaln = AdaLn(action_expert_config.width)

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if not os.environ.get("TORCHDYNAMO_DISABLE"):
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None
        
    def set_video_latent_shape(self, video_latent_shape):
        self.video_latent_shape = video_latent_shape

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.gemma_expert.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.gemma_expert.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        return observation.state

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, video_embeddings, timestep
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer 
        """

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        video_embeds = self.video_adaln(video_embeddings, time_emb)
        embs = [video_embeds]
        bsize, num_video_embeds = video_embeddings.shape[:2]

        embs = torch.cat(embs, dim=1)

        video_masks = torch.ones((bsize,), dtype=torch.bool, device=video_embeddings.device)
        pad_masks = video_masks[:, None].expand(bsize, num_video_embeds)
        att_masks = [0] * num_video_embeds
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Cast all inputs to model dtype
        model_dtype = self.state_proj.weight.dtype
        state = state.to(model_dtype)
        noisy_actions = noisy_actions.to(model_dtype)
        timestep = timestep.to(model_dtype)

        # Embed state
        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)

        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # time MLP (for adaRMS)
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)  # swish == silu
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def patchify_video_latents(self, video_latents):
        """
        Convert video latents to patches for transformer input.
        video latents: (batch_size, seq_len, hidden_dim)
        First need to reshape to (batch_size, temporal, height, width, hidden_dim)
        Then zero-pad spatial dims to even and convert to 2x2 patches.
        """

        bsize, seq_len, hidden_dim = video_latents.shape
        T, H, W = self.video_latent_shape
        video_latents = video_latents.view(bsize, T, H, W, hidden_dim)

        # Pad spatial dims to even for 2x2 patchify
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            video_latents = torch.nn.functional.pad(video_latents, (0, 0, 0, pad_w, 0, pad_h))

        num_latent_frames = video_latents.shape[1]

        video_latents = rearrange(video_latents, 'b s (h p1) (w p2) c -> b s h w (p1 p2 c)', p1=2, p2=2)
        video_latents = video_latents.view(bsize, num_latent_frames, -1, hidden_dim * 4)
        video_latents = self.video_latent_embedder(video_latents)
        video_latents = video_latents.view(bsize, -1, self.action_expert_width)

        return video_latents, num_latent_frames

    def preprocess_all_chunks(self, prefix_embeds, suffix_embeds, prefix_pad_masks, suffix_pad_masks, num_latent_frames):

        bsize, prefix_len, _ = prefix_embeds.shape
        _, suffix_len, _ = suffix_embeds.shape

        num_spatial_tokens = prefix_len // num_latent_frames  # assuming num_latent_frames temporal tokens
        prefix_embeds = prefix_embeds.view(bsize, num_latent_frames, num_spatial_tokens, -1) # (bsize, 5, num_spatial_tokens, dim)
        num_state_tokens = num_spatial_tokens
        state_embeds = suffix_embeds[:, :1, :].unsqueeze(1).repeat(1, 1, num_state_tokens, 1) # (bsize, 1, 1, dim)
        action_embeds = suffix_embeds[:, 1:, :]

        prefix_pad_masks = prefix_pad_masks.view(bsize, num_latent_frames, num_spatial_tokens)  # (bsize, 5, num_spatial_tokens)
        state_pad_masks = suffix_pad_masks[:, :1].unsqueeze(1).repeat(1, 1, num_state_tokens) # (bsize, 1, num_spatial_tokens)
        action_pad_masks = suffix_pad_masks[:, 1:]  # (bsize, 1, suffix_len-1)

        interleaved_sequence = []
        interleaved_masks = []
        action_indices = []
        seq_ptr = 0
        for t in range(num_latent_frames):
            interleaved_sequence.append(prefix_embeds[:, t, :, :].view(bsize, num_spatial_tokens, -1))  # spatial tokens at time t
            interleaved_masks.append(prefix_pad_masks[:, t, :])  # corresponding pad masks
            seq_ptr += num_spatial_tokens
            if t == 0:
                interleaved_sequence.append(state_embeds[:, 0, :, :].view(bsize, num_state_tokens, -1))  # state tokens after first temporal token
                interleaved_masks.append(state_pad_masks[:, 0, :])  # corresponding pad masks
                seq_ptr += num_state_tokens

                interleaved_sequence.append(action_embeds[:, :1, :])  # first action token after state
                interleaved_masks.append(action_pad_masks[:, :1])  # corresponding pad masks
                seq_ptr += 1
                action_indices.append(seq_ptr -1)
            else:
                interleaved_sequence.append(action_embeds[:, (t-1)*4 +1 : t*4 +1, :])  # next 4 action tokens after each temporal token
                interleaved_masks.append(action_pad_masks[:, (t-1)*4 +1 : t*4 +1])  # corresponding pad masks
                seq_ptr += 4
                action_indices += list(range(seq_ptr - 4, seq_ptr))

        interleaved_embeds = torch.cat(interleaved_sequence, dim=1)  # (bsize, total_len, dim)
        interleaved_pad_masks = torch.cat(interleaved_masks, dim=1)  # (bsize, total_len)

        assert len(action_indices) == suffix_len - 1, "Action indices length mismatch"

        return interleaved_embeds, interleaved_pad_masks, action_indices

    def get_interleaved_masks_multimodal(self, x, action_indices, t_idx):
        bsize, total_len, _ = x.shape
        device = x.device
        
        # Ensure t_idx is 1D for the outer product math: [1, seq] -> [seq]
        if t_idx.dim() == 2:
            t_idx = t_idx.squeeze(0)

        # 1. Temporal Indices outer product -> [seq, seq]
        t_row = t_idx.unsqueeze(1) # [605, 1]
        t_col = t_idx.unsqueeze(0) # [1, 605]
        
        # 2. Block Causal Base: Observer sees Target if Target_T <= Observer_T
        mask_2d = (t_row >= t_col)

        # 3. Action Sequence Causality
        sequential_causal = torch.tril(torch.ones(total_len, total_len, device=device, dtype=torch.bool))
        
        is_action = torch.zeros(total_len, dtype=torch.bool, device=device)
        is_action[action_indices] = True
        is_action_row = is_action.view(-1, 1)
        is_action_col = is_action.view(1, -1)

        # Rule 1: Actions follow strict 1D sequence
        action_to_action_future = is_action_row & is_action_col & (~sequential_causal)
        mask_2d = mask_2d & (~action_to_action_future)

        # Rule 2: Video Frames cannot see Actions
        video_sees_action = (~is_action_row) & is_action_col
        mask_2d = mask_2d & (~video_sees_action)

        # 4. Final Gemma Format
        causal_mask = torch.full((bsize, 1, total_len, total_len), float("-inf"), device=device)
        causal_mask.masked_fill_(mask_2d.unsqueeze(0).unsqueeze(1), 0.0)
        
        return causal_mask

    def _padded_patch_dims(self):
        """Return (V_T, V_H, V_W) after padding to even and 2x2 patchify."""
        T, H, W = self.video_latent_shape
        return T, (H + H % 2) // 2, (W + W % 2) // 2

    def get_mrope_triplet(self, device):
        V_T, V_H, V_W = self._padded_patch_dims()
        num_spatial_tokens = V_H * V_W # 98
        
        t_ids, h_ids, w_ids = [], [], []

        # 1. Base spatial grid for Frame/State
        grid_h, grid_w = torch.meshgrid(
            torch.arange(V_H, device=device), 
            torch.arange(V_W, device=device), 
            indexing='ij'
        )
        grid_h = grid_h.reshape(-1) # [98]
        grid_w = grid_w.reshape(-1) # [98]

        # 2. Replicate your Interleaving Logic
        for t in range(V_T):
            # --- VIDEO FRAME t ---
            # 1+4n rule: 0, 1, 5, 9, 13
            t_val = 0 if t == 0 else (t-1) * 4 + 1
            t_ids.append(torch.full((num_spatial_tokens,), t_val, device=device))
            h_ids.append(grid_h)
            w_ids.append(grid_w)
            
            if t == 0:
                # --- STATE FRAME (98 tokens) ---
                t_ids.append(torch.full((num_spatial_tokens,), 0, device=device))
                h_ids.append(grid_h) # Spatially aligned with Frame 0
                w_ids.append(grid_w)
                
                # --- ACTION 0 (1 token) ---
                t_ids.append(torch.tensor([0], device=device))
                h_ids.append(torch.tensor([0], device=device))
                w_ids.append(torch.tensor([0], device=device))
            else:
                # --- ACTION CHUNKS (4 tokens) ---
                # Actions 1-4, 5-8, etc. share the T of their frame
                chunk_t = torch.full((4,), t_val, device=device)
                t_ids.append(chunk_t)
                h_ids.append(torch.zeros(4, device=device))
                w_ids.append(torch.zeros(4, device=device))

        # 3. Final Concatenation
        T = torch.cat(t_ids).unsqueeze(0) # [1, 604]
        H = torch.cat(h_ids).unsqueeze(0) # [1, 604]
        W = torch.cat(w_ids).unsqueeze(0) # [1, 604]
        
        return T.long(), H.long(), W.long()
        
    def get_interleaved_masks(self, x):
        # 1. Base Causal Mask
        bsize, total_len, _ = x.shape
        device = x.device
        causal_mask = torch.full((total_len, total_len), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        
        # 2. Add Batch and Head dimensions for Gemma
        # [1, 1, Total_Len, Total_Len]
        causal_mask = causal_mask[None, None, :, :]
        
        # 3. Expand to Batch Size
        # Note: Gemma often handles the head expansion internally
        return causal_mask.expand(bsize, -1, -1, -1)

    def get_position_ids(self, x):
        V_T, V_H, V_W = self._padded_patch_dims()
        MAX_H, MAX_W = V_H, V_W
        num_spatial_tokens = V_H * V_W # 98
        
        # 1. Pre-calculate the spatial grid
        grid_h, grid_w = torch.meshgrid(
            torch.arange(V_H, device=x.device), 
            torch.arange(V_W, device=x.device), 
            indexing='ij'
        )
        grid_h = grid_h.reshape(-1) # [98]
        grid_w = grid_w.reshape(-1) # [98]
        # Spatial base for 1D flattening: (H * MAX_W) + W
        spatial_flat = grid_h * MAX_W + grid_w

        t_ids = []

        for t in range(V_T):
            # --- VIDEO FRAME t ---
            # 1+4n rule: 0, 1, 5, 9, 13
            t_val = 0 if t == 0 else (t-1) * 4 + 1
            
            # Flattened Index: (T * H_max * W_max) + Spatial_Flat
            frame_ids = (t_val * MAX_H * MAX_W) + spatial_flat
            t_ids.append(frame_ids)
            
            if t == 0:
                # --- STATE FRAME (Repeated 98 times) ---
                # Anchored at T=0. We give it the same spatial IDs as Frame 0.
                state_frame_ids = (0 * MAX_H * MAX_W) + spatial_flat
                t_ids.append(state_frame_ids)
                
                # --- ACTION 0 ---
                # T=0, H=0, W=0 -> Index 0
                t_ids.append(torch.tensor([0], device=x.device))
            else:
                # --- ACTION CHUNKS ---
                # Actions 1-4, 5-8, etc. 
                # We treat actions as spatial 0: (T * H_max * W_max) + 0 + 0
                chunk_t = torch.arange((t-1)*4 + 1, t*4 + 1, device=x.device)
                action_ids = chunk_t * MAX_H * MAX_W
                t_ids.append(action_ids)

        position_ids = torch.cat(t_ids).unsqueeze(0) # [1, 604]
        return position_ids.to(dtype=torch.long, device=x.device)

    def forward(self, observation, actions, video_latents, video_timesteps, noise=None, time=None,
                state_dropout_rate: float = 0.0, first_frame_drop_prob: float = 0.0) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        state = self._preprocess_observation(observation, train=True)

        if self.training and state_dropout_rate > 0:
            mask = torch.rand(state.shape[0], 1, device=state.device) > state_dropout_rate
            state = state * mask.to(state.dtype)

        video_latents, num_latent_frames = self.patchify_video_latents(video_latents)

        if self.action_horizon is None:
            self.action_horizon = actions.shape[1]

        video_timesteps = video_timesteps.to(dtype=actions.dtype, device=actions.device)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(video_latents, video_timesteps)
        if self.training and first_frame_drop_prob > 0:
            if self.interleaved:
                raise NotImplementedError("first_frame_drop_prob is only supported by the non-interleaved layout")
            prefix_pad_masks = drop_first_frame_tokens(prefix_pad_masks, num_latent_frames, first_frame_drop_prob)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        if self.interleaved:
            interleaved_suffix_embs, interleaved_suffix_pad_masks, action_indices = self.preprocess_all_chunks(
                prefix_embs, suffix_embs, prefix_pad_masks, suffix_pad_masks, num_latent_frames
            )

            assert len(action_indices) == self.config.action_horizon, "action horizon mismatch"

            if self.multimodal:
                position_ids = self.get_mrope_triplet(interleaved_suffix_embs.device)
                att_2d_masks_4d = self.get_interleaved_masks(interleaved_suffix_embs, action_indices, position_ids[0]).to(dtype=torch.bfloat16)
            else:
                att_2d_masks_4d = self.get_interleaved_masks(interleaved_suffix_embs).to(dtype=torch.bfloat16)
                position_ids = self.get_position_ids(interleaved_suffix_embs)
            combined_embeds = interleaved_suffix_embs
        else:
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            # Prepare attention masks
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=torch.bfloat16)
            combined_embeds = torch.cat([prefix_embs, suffix_embs], dim=1)

        # Apply gradient checkpointing if enabled
        def forward_func(combined_embeds, att_2d_masks_4d, position_ids, adarms_cond):
            suffix_out = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=combined_embeds,
                use_cache=False,
                adarms_cond=adarms_cond,
                action_indices=action_indices if self.multimodal else None,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, combined_embeds, att_2d_masks_4d, position_ids, adarms_cond
        )

        if self.interleaved:
            suffix_out = suffix_out[:, action_indices]
        else:
            suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        loss = F.mse_loss(u_t, v_t, reduction="none")
        if loss.mean() > 1e6:
            import logging as _log
            _log.warning(
                f"[ACTION_DEBUG] loss={loss.mean():.2e} "
                f"u_t=[{u_t.min():.3f},{u_t.max():.3f}] "
                f"v_t=[{v_t.min():.3f},{v_t.max():.3f}] "
                f"actions=[{actions.min():.3f},{actions.max():.3f}] "
                f"state=[{state.min():.3f},{state.max():.3f}] "
                f"time=[{time.min():.3f},{time.max():.3f}]"
            )
        return loss

    @torch.no_grad()
    def sample_actions(self, device, observation, video_latents, video_timestep, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        state = self._preprocess_observation(observation, train=False)

        video_latents, num_latent_frames = self.patchify_video_latents(video_latents)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(video_latents, video_timestep)
        if self.multimodal:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, expanded_time)

            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            if self.interleaved:
                interleaved_suffix_embs, interleaved_suffix_pad_masks, action_indices = self.preprocess_all_chunks(
                    prefix_embs, suffix_embs, prefix_pad_masks, suffix_pad_masks, num_latent_frames
                )

                if self.multimodal:
                    position_ids = self.get_mrope_triplet(interleaved_suffix_embs.device)
                    att_2d_masks_4d = self.get_interleaved_masks(interleaved_suffix_embs, action_indices, position_ids[0]).to(dtype=torch.bfloat16)
                else:
                    att_2d_masks_4d = self.get_interleaved_masks(interleaved_suffix_embs).to(dtype=torch.bfloat16)
                    position_ids = self.get_position_ids(interleaved_suffix_embs)
                combined_embeds = interleaved_suffix_embs
            else:
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1

                # Prepare attention masks
                att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=torch.bfloat16)
                combined_embeds = torch.cat([prefix_embs, suffix_embs], dim=1)

            self.paligemma_with_expert.gemma_expert.config._attn_implementation = "eager"  # noqa: SLF001
            suffix_out = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=combined_embeds,
                use_cache=False,
                adarms_cond=adarms_cond,
                action_indices=action_indices if self.multimodal else None,
            )

            if self.interleaved:
                suffix_out = suffix_out[:, action_indices]
            else:
                suffix_out = suffix_out[:, -self.action_horizon :]
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out).float()

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def sample_actions_cfg(self, device, observation, video_latents_cond, video_timestep_cond,
                           video_latents_uncond, video_timestep_uncond,
                           cfg_weight=2.0, noise=None, num_steps=10) -> Tensor:
        """Flow-matching inference with classifier-free guidance on the velocity.

        At each Euler step:  v = v_uncond + cfg_weight * (v_cond - v_uncond)
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        state = self._preprocess_observation(observation, train=False)

        vl_cond, nlf = self.patchify_video_latents(video_latents_cond)
        prefix_cond, pp_cond, pa_cond = self.embed_prefix(vl_cond, video_timestep_cond)
        vl_uncond, _ = self.patchify_video_latents(video_latents_uncond)
        prefix_uncond, pp_uncond, pa_uncond = self.embed_prefix(vl_uncond, video_timestep_uncond)

        if self.multimodal:
            prefix_cond = prefix_cond.to(dtype=torch.bfloat16)
            prefix_uncond = prefix_uncond.to(dtype=torch.bfloat16)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        def _forward_once(prefix_embs, prefix_pad_masks, prefix_att_masks):
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, expanded_time)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            pe = prefix_embs.to(dtype=torch.bfloat16)

            if self.interleaved:
                interleaved, interleaved_pad, action_indices = self.preprocess_all_chunks(
                    pe, suffix_embs, prefix_pad_masks, suffix_pad_masks, nlf)
                if self.multimodal:
                    position_ids = self.get_mrope_triplet(interleaved.device)
                    att_4d = self.get_interleaved_masks(interleaved, action_indices, position_ids[0]).to(dtype=torch.bfloat16)
                else:
                    att_4d = self.get_interleaved_masks(interleaved).to(dtype=torch.bfloat16)
                    position_ids = self.get_position_ids(interleaved)
                combined = interleaved
            else:
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
                att_2d = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1
                att_4d = self._prepare_attention_masks_4d(att_2d).to(dtype=torch.bfloat16)
                combined = torch.cat([pe, suffix_embs], dim=1)

            self.paligemma_with_expert.gemma_expert.config._attn_implementation = "eager"
            out = self.paligemma_with_expert.forward(
                attention_mask=att_4d, position_ids=position_ids,
                past_key_values=None, inputs_embeds=combined, use_cache=False,
                adarms_cond=adarms_cond,
                action_indices=action_indices if self.multimodal else None,
            )
            if self.interleaved:
                out = out[:, action_indices]
            else:
                out = out[:, -self.action_horizon:]
            out = out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(out).float()

        while time >= -dt / 2:
            v_cond = _forward_once(prefix_cond, pp_cond, pa_cond)
            v_uncond = _forward_once(prefix_uncond, pp_uncond, pa_uncond)
            v_t = v_uncond + cfg_weight * (v_cond - v_uncond)
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def sample_actions_adaptive_cfg(
        self, device, observation,
        video_latents_cond, video_timestep_cond,
        video_latents_uncond=None, video_timestep_uncond=None,
        video_latents_plan=None, video_timestep_plan=None,
        w_lang: float = 0.0,
        w_plan: float = 0.0,
        noise=None, num_steps=10,
        tr_layer: int | None = None,
        cfg_formula: str = "extrapolation",
        tr_all_layers: bool = False,
    ) -> tuple[Tensor, float | dict[int, float]]:
        """Flow-matching with velocity-space adaptive CFG + TR capture.

        At each Euler step:
            v = v_cond + w_lang*(v_cond - v_uncond) + w_plan*(v_plan - v_cond)

        TR is computed from attention on the FIRST denoising step only.

        Returns:
            (actions, tr): denoised actions and temporal ratio from step 0 -- of ``tr_layer``, or
            ``{layer: tr}`` for every action-expert layer when ``tr_all_layers`` is set.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        state = self._preprocess_observation(observation, train=False)

        vl_cond, nlf = self.patchify_video_latents(video_latents_cond)
        prefix_cond, pp_cond, pa_cond = self.embed_prefix(vl_cond, video_timestep_cond)

        has_uncond = video_latents_uncond is not None and w_lang > 0.01
        has_plan = video_latents_plan is not None and w_plan > 0.01

        if has_uncond:
            vl_uncond, _ = self.patchify_video_latents(video_latents_uncond)
            prefix_uncond, pp_uncond, pa_uncond = self.embed_prefix(vl_uncond, video_timestep_uncond)
        if has_plan:
            vl_plan, _ = self.patchify_video_latents(video_latents_plan)
            prefix_plan, pp_plan, pa_plan = self.embed_prefix(vl_plan, video_timestep_plan)

        if self.multimodal:
            prefix_cond = prefix_cond.to(dtype=torch.bfloat16)
            if has_uncond:
                prefix_uncond = prefix_uncond.to(dtype=torch.bfloat16)
            if has_plan:
                prefix_plan = prefix_plan.to(dtype=torch.bfloat16)

        dt = -1.0 / num_steps
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time_val = torch.tensor(1.0, dtype=torch.float32, device=device)

        num_spatial_tokens = prefix_cond.shape[1] // nlf

        self.paligemma_with_expert.gemma_expert.config._attn_implementation = "eager"

        def _forward_once(prefix_embs, prefix_pad_masks, prefix_att_masks):
            expanded_time = time_val.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, expanded_time)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            pe = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(att_2d).to(dtype=torch.bfloat16)
            combined = torch.cat([pe, suffix_embs], dim=1)

            out = self.paligemma_with_expert.forward(
                attention_mask=att_4d, position_ids=position_ids,
                past_key_values=None, inputs_embeds=combined, use_cache=False,
                adarms_cond=adarms_cond,
            )
            out = out[:, -self.action_horizon:]
            out = out.to(dtype=self.action_out_proj.weight.dtype)
            return self.action_out_proj(out).float(), combined.shape[1]

        tr = 1.0
        step = 0
        while time_val >= -dt_t / 2:
            capture_tr = (step == 0)
            if capture_tr:
                layers_to_hook = [tr_layer] if tr_layer is not None else None
                handles, storage = self._register_attn_hooks()

            v_cond, total_len = _forward_once(prefix_cond, pp_cond, pa_cond)

            if capture_tr:
                for h in handles:
                    h.remove()
                if tr_all_layers:
                    tr = {
                        layer: self._compute_tr_from_storage(storage, nlf, num_spatial_tokens, total_len, layer)
                        for layer in sorted(storage)
                    }
                else:
                    tr = self._compute_tr_from_storage(
                        storage, nlf, num_spatial_tokens, total_len, tr_layer
                    )

            if has_uncond:
                v_uncond, _ = _forward_once(prefix_uncond, pp_uncond, pa_uncond)
            if has_plan:
                v_plan, _ = _forward_once(prefix_plan, pp_plan, pa_plan)

            if cfg_formula == "standard" and has_uncond:
                v_t = v_uncond + w_lang * (v_cond - v_uncond)
            else:
                v_t = v_cond.clone()
                if has_uncond:
                    v_t = v_t + w_lang * (v_cond - v_uncond)
            if has_plan:
                v_t = v_t + w_plan * (v_plan - v_cond)

            x_t = x_t + dt_t * v_t
            time_val += dt_t
            step += 1

        return x_t, tr

    def _compute_tr_from_storage(self, storage, num_latent_frames, num_spatial_tokens,
                                  total_len, tr_layer=None):
        """Extract temporal ratio from captured attention weights."""
        segments = self._build_segment_map(num_latent_frames, num_spatial_tokens, total_len)
        seg_names = list(segments.keys())
        video_indices = [i for i, name in enumerate(seg_names) if name.startswith("video_t")]
        if len(video_indices) < 2:
            return 1.0

        action_indices = list(range(total_len - self.action_horizon, total_len))

        if tr_layer is not None and tr_layer in storage:
            layers = [tr_layer]
        else:
            layers = sorted(storage.keys())

        w = storage[layers[0]]  # (B, heads, S, S)
        w0 = w[0].mean(dim=0)  # (S, S) — average over heads
        action_attn = w0[action_indices]  # (n_actions, S)

        t0_mass = sum(action_attn[:, idx].sum().item() for idx in segments[seg_names[video_indices[0]]])
        future_mass = 0.0
        for vi in video_indices[1:]:
            future_mass += sum(action_attn[:, idx].sum().item() for idx in segments[seg_names[vi]])

        return float(future_mass / (t0_mass + 1e-8))

    # ------------------------------------------------------------------
    # Self-attention visualizer: which tokens do action tokens attend to?
    # ------------------------------------------------------------------

    def _build_segment_map(self, num_latent_frames, num_spatial_tokens, total_len):
        """Build an ordered map of segment_name -> list of token indices.

        For the non-interleaved layout the sequence is:
            [video_t0 | video_t1 | ... | video_tN | state | action_0 ... action_H]

        Returns:
            OrderedDict mapping segment names to their index ranges.
        """
        segments = OrderedDict()
        ptr = 0
        for t in range(num_latent_frames):
            segments[f"video_t{t}"] = list(range(ptr, ptr + num_spatial_tokens))
            ptr += num_spatial_tokens
        # suffix: 1 state token + action_horizon action tokens
        segments["state"] = [ptr]
        ptr += 1
        segments["actions"] = list(range(ptr, total_len))
        return segments

    def _register_attn_hooks(self):
        """Register forward hooks on every GemmaAttention to capture weights.

        Returns (hook_handles, storage_dict).  Caller must remove hooks when done.
        """
        storage = {}
        handles = []
        for layer_idx, layer in enumerate(
            self.paligemma_with_expert.gemma_expert.layers
        ):
            def _make_hook(idx):
                def _hook(module, _input, output):
                    # GemmaAttention.forward returns (attn_output, attn_weights)
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        # attn_weights: (B, n_heads, S, S)
                        storage[idx] = output[1].detach().cpu().float()
                return _hook
            handles.append(layer.self_attn.register_forward_hook(_make_hook(layer_idx)))
        return handles, storage

    @torch.no_grad()
    def visualize_attention(
        self,
        device,
        observation,
        video_latents,
        video_timestep,
        noise=None,
        denoise_step_idx=0,
        num_steps=10,
        save_path=None,
        layers=None,
        head_reduction="mean",
    ):
        """Run inference and visualize which tokens action tokens attend to.

        Works with the non-interleaved path (interleaved=False, multimodal=False).
        Sequence layout: [video_t0 | video_t1 | ... | state | action_0 ... action_H]

        Args:
            device, observation, video_latents, video_timestep, noise, num_steps:
                Same as sample_actions.
            denoise_step_idx: Which denoising step to capture (0 = first, i.e.
                pure noise input).  Use -1 for the last step.
            save_path: If given, save the figure to this path (e.g. "attn.png").
                If None, calls plt.show().
            layers: List of layer indices to visualize, or None for all layers.
            head_reduction: How to reduce across heads — "mean" or "max".

        Returns:
            dict with keys:
                "segment_attention": np.ndarray (n_layers, n_actions, n_segments)
                    — attention mass each action token places on each segment,
                    averaged/maxed across heads.
                "segment_names": list[str]
                "action_indices": list[int]
                "raw_weights": dict[layer_idx -> np.ndarray (B, heads, S, S)]
                    — full attention weight matrices (only captured step).
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        state = self._preprocess_observation(observation, train=False)
        video_latents_patched, num_latent_frames = self.patchify_video_latents(video_latents)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            video_latents_patched, video_timestep
        )
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        num_spatial_tokens = prefix_embs.shape[1] // num_latent_frames

        dt = -1.0 / num_steps
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time_val = torch.tensor(1.0, dtype=torch.float32, device=device)
        step = 0
        captured_storage = None
        total_len = None

        if denoise_step_idx < 0:
            denoise_step_idx = num_steps + denoise_step_idx

        while time_val >= -dt_t / 2:
            expanded_time = time_val.expand(bsize)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, expanded_time
            )
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=torch.bfloat16)
            combined_embeds = torch.cat([prefix_embs, suffix_embs], dim=1)
            total_len = combined_embeds.shape[1]

            # Force eager attention so we get weights
            self.paligemma_with_expert.gemma_expert.config._attn_implementation = "eager"  # noqa: SLF001

            # Register hooks only on the target step
            if step == denoise_step_idx:
                handles, storage = self._register_attn_hooks()

            suffix_out = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=combined_embeds,
                use_cache=False,
                adarms_cond=adarms_cond,
            )

            if step == denoise_step_idx:
                for h in handles:
                    h.remove()
                captured_storage = storage

            suffix_out = suffix_out[:, -self.action_horizon:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt_t * v_t
            time_val += dt_t
            step += 1

        # ----- Build segment map and aggregate attention -----
        # Action tokens are the last action_horizon positions
        action_token_indices = list(range(total_len - self.action_horizon, total_len))
        segments = self._build_segment_map(num_latent_frames, num_spatial_tokens, total_len)
        seg_names = list(segments.keys())
        n_layers = len(captured_storage)
        n_actions = len(action_token_indices)
        n_segments = len(seg_names)

        if layers is None:
            layers = sorted(captured_storage.keys())

        reduce_fn = torch.mean if head_reduction == "mean" else torch.amax

        # segment_attention: (len(layers), n_actions, n_segments)
        seg_attn = np.zeros((len(layers), n_actions, n_segments), dtype=np.float32)
        for li, layer_idx in enumerate(layers):
            w = captured_storage[layer_idx]  # (B, heads, S, S)
            # Take first batch element, reduce heads
            w0 = reduce_fn(w[0], dim=0)  # (S, S)
            for ai, a_idx in enumerate(action_token_indices):
                row = w0[a_idx]  # (S,)
                for si, seg_name in enumerate(seg_names):
                    seg_attn[li, ai, si] = row[segments[seg_name]].sum().item()

        raw_np = {k: v.numpy() for k, v in captured_storage.items()}

        # ----- Plot (only when save_path is provided) -----
        if HAS_MATPLOTLIB and save_path:
            n_plot_layers = len(layers)
            fig, axes = plt.subplots(
                1, n_plot_layers,
                figsize=(5 * n_plot_layers + 2, max(4, n_actions * 0.35)),
                squeeze=False,
            )
            for li, layer_idx in enumerate(layers):
                ax = axes[0, li]
                im = ax.imshow(seg_attn[li], aspect="auto", cmap="viridis")
                ax.set_title(f"Layer {layer_idx}")
                ax.set_xlabel("Segment")
                ax.set_ylabel("Action token")
                ax.set_xticks(range(n_segments))
                ax.set_xticklabels(seg_names, rotation=60, ha="right", fontsize=7)
                ax.set_yticks(range(n_actions))
                ax.set_yticklabels(
                    [f"a{i}" for i in range(n_actions)], fontsize=7
                )
            fig.colorbar(im, ax=axes.ravel().tolist(), label="Attention mass", shrink=0.8)
            fig.suptitle(
                f"Self-attention from action tokens (step {denoise_step_idx}, "
                f"head {head_reduction})",
                fontsize=11,
            )
            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logging.info(f"Saved attention visualization to {save_path}")
            plt.close(fig)

        return {
            "actions": x_t,  # denoised actions: (B, action_horizon, action_dim)
            "segment_attention": seg_attn,
            "segment_names": seg_names,
            "action_token_indices": action_token_indices,
            "raw_weights": raw_np,
        }