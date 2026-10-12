"""
Self-contained Reason1 text encoder for Cosmos Predict 2.5.

Loads nvidia/Cosmos-Reason1-7B (fine-tuned Qwen2.5-VL-7B) from HuggingFace
and produces text embeddings compatible with the Cosmos DiT cross-attention.

Strategy: FULL_CONCAT — concatenates hidden states from all 28 transformer
layers, each 3584-dim, giving (B, 512, 100352) embeddings. The DiT's
crossattn_proj then projects 100352 → 1024 before cross-attention.

No dependency on cosmos_predict2.

Usage:
    encoder = Reason1TextEncoder(device="cuda:0")
    embeddings = encoder.encode("Pick up the red block.")
    # embeddings: (1, 512, 100352) bfloat16
"""

import logging
from typing import Optional

import torch

_log = logging.getLogger(__name__)

# Padding length for tokenized sequences (matches cosmos_predict2 TextEncoder)
NUM_EMBEDDING_PADDING_TOKENS = 512

# Hugging Face model and the revision pinned by the Cosmos checkpoint registry
# (cosmos_video/_src/imaginaire/utils/checkpoint_db.py), so that both share one cached copy.
REASON1_MODEL_ID = "nvidia/Cosmos-Reason1-7B"
REASON1_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"


class Reason1TextEncoder:
    """Self-contained Reason1 (Qwen2.5-VL-7B) text encoder.

    Produces (B, 512, 100352) bfloat16 embeddings via full_concat of all
    28 hidden layers (each 3584-dim).
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        model_id: str = REASON1_MODEL_ID,
        dtype: torch.dtype = torch.bfloat16,
        revision: str | None = REASON1_REVISION,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        _log.info(f"Loading Reason1 text encoder from {model_id}")

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map=str(self.device),
            output_hidden_states=True,
        )
        self.model.eval()

        # Cache the pad token id
        self.pad_id = self.processor.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.processor.tokenizer.eos_token_id

        _log.info(
            f"Reason1 loaded: {sum(p.numel() for p in self.model.parameters()):,} params, "
            f"hidden_size=3584, num_layers=28"
        )

    @staticmethod
    def _mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Mean-normalize along the last dimension."""
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt using the Qwen2.5-VL chat template.

        Returns input_ids of length NUM_EMBEDDING_PADDING_TOKENS (512),
        padded or truncated as needed.
        """
        conversations = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant who will provide prompts to an image generator."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        text = self.processor.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = self.processor.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0).tolist()

        # Pad or truncate to fixed length
        if len(input_ids) < NUM_EMBEDDING_PADDING_TOKENS:
            input_ids = input_ids + [self.pad_id] * (NUM_EMBEDDING_PADDING_TOKENS - len(input_ids))
        else:
            input_ids = input_ids[:NUM_EMBEDDING_PADDING_TOKENS]

        return torch.tensor(input_ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def encode(
        self,
        prompts: str | list[str],
    ) -> torch.Tensor:
        """Encode prompt(s) to Reason1 embeddings.

        Args:
            prompts: Single prompt string or list of prompts.

        Returns:
            Tensor of shape (B, 512, 100352) in bfloat16, where
            100352 = 28 layers × 3584 hidden_size (full_concat).
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize all prompts
        input_ids_batch = torch.stack([self._tokenize_prompt(p) for p in prompts])

        # Forward pass with hidden states
        outputs = self.model(
            input_ids=input_ids_batch,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors, each (B, seq, 3584)

        # Full concat of all hidden layers (skip embedding layer at index 0)
        normalized = []
        for layer_idx in range(1, len(hidden_states)):
            normalized.append(self._mean_normalize(hidden_states[layer_idx]))

        # Concatenate along feature dimension: (B, 512, 28*3584) = (B, 512, 100352)
        text_embeddings = torch.cat(normalized, dim=-1)

        return text_embeddings.to(self.dtype)

    def offload_to_cpu(self):
        """Move model to CPU to free GPU memory."""
        self.model = self.model.to("cpu")
        torch.cuda.empty_cache()

    def reload_to_device(self):
        """Move model back to the original device."""
        self.model = self.model.to(self.device)
