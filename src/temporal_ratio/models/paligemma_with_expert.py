from typing import Literal

import torch
from torch import nn
from transformers import GemmaModel
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.cache_utils import Cache
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        multimodal=False,
    ):
        if use_adarms is None:
            use_adarms = False
        super().__init__()

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms,
            adarms_cond_dim=action_expert_config.width if use_adarms else None,
        )

        self.multimodal = multimodal
        self.gemma_expert = GemmaModel(config=action_expert_config_hf) if not multimodal else GemmaModel(config=action_expert_config_hf)
        self.gemma_expert.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor] | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        adarms_cond: torch.Tensor | None = None,
        action_indices: list[int] | None = None,
    ):
        if self.multimodal:
            suffix_output = self.gemma_expert.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond if adarms_cond is not None else None,
                action_indices=action_indices if action_indices is not None else None,
            )
        else:
            suffix_output = self.gemma_expert.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond if adarms_cond is not None else None,
            )
        suffix_output = suffix_output.last_hidden_state

        return suffix_output
