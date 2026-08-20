"""Looped Qwen3 built by reusing Hugging Face's official Qwen3 components.

Only the traversal of decoder layers is changed: the same ModuleList is
applied repeatedly. Attention, Q/K normalization, RoPE, MLP, masks, loss,
initialization and PreTrainedModel integration come from Transformers.
"""

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)


class LoopedQwen3Config(Qwen3Config):
    model_type = "looped_qwen3"

    def __init__(self, num_loops: int = 4, **kwargs: Any) -> None:
        self.num_loops = num_loops
        super().__init__(**kwargs)
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive")
        if self.use_cache:
            # A standard per-layer Qwen3 KV cache is not loop-aware.
            self.use_cache = False


class LoopedQwen3Model(Qwen3Model):
    """Official Qwen3Model with its layer stack recurrently reused."""

    config_class = LoopedQwen3Config

    def __init__(self, config: LoopedQwen3Config) -> None:
        super().__init__(config)
        self.last_relative_updates: tuple[torch.Tensor, ...] = ()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        num_loops: int | None = None,
        return_loop_diagnostics: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache or past_key_values is not None:
            raise ValueError(
                "KV cache is disabled: the standard Qwen3 cache is not loop-aware. "
                "Use use_cache=False."
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        loops = self.config.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError("num_loops must be positive")
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        relative_updates = []

        for _ in range(loops):
            previous = hidden_states
            for layer_idx, decoder_layer in enumerate(self.layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    **kwargs,
                )
            if return_loop_diagnostics:
                numerator = (hidden_states.float() - previous.float()).norm(dim=-1).mean()
                denominator = previous.float().norm(dim=-1).mean().clamp_min(1e-8)
                relative_updates.append((numerator / denominator).detach())

        self.last_relative_updates = tuple(relative_updates)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)


class LoopedQwen3ForCausalLM(Qwen3ForCausalLM):
    """Causal LM wrapper retaining the official Qwen3 loss and HF APIs."""

    config_class = LoopedQwen3Config

    def __init__(self, config: LoopedQwen3Config) -> None:
        # Avoid constructing and discarding the ordinary Qwen3Model created by
        # Qwen3ForCausalLM.__init__.
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = LoopedQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        num_loops: int | None = None,
        return_loop_diagnostics: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        # This mirrors the thin official Qwen3ForCausalLM wrapper. Custom loop
        # arguments are consumed here instead of leaking into the HF loss.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            num_loops=num_loops,
            return_loop_diagnostics=return_loop_diagnostics,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)


# Local AutoClass support and portable save_pretrained() code packaging.
try:
    AutoConfig.register(LoopedQwen3Config.model_type, LoopedQwen3Config)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(LoopedQwen3Config, LoopedQwen3ForCausalLM)
except ValueError:
    pass
LoopedQwen3Config.register_for_auto_class()
LoopedQwen3ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
