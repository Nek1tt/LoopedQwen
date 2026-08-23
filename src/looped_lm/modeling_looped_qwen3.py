"""Looped Qwen3 built by reusing Hugging Face's official Qwen3 components.

Only the traversal of decoder layers is changed: the same ModuleList is
applied repeatedly. Attention, Q/K normalization, RoPE, MLP, masks, loss,
initialization and PreTrainedModel integration come from Transformers.
"""

import math
from dataclasses import dataclass
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


@dataclass
class LoopedCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal-LM output with the per-depth losses used by anytime training."""

    loop_losses: torch.FloatTensor | None = None
    loop_uniform_losses: torch.FloatTensor | None = None
    loop_corrective_losses: torch.FloatTensor | None = None
    loop_hard_weight_means: torch.FloatTensor | None = None
    supervised_loops: tuple[int, ...] | None = None


class LoopedQwen3Config(Qwen3Config):
    model_type = "looped_qwen3"

    def __init__(
        self,
        num_loops: int = 4,
        loop_update_mode: str = "full",
        loop_update_alpha: float = 1.0,
        loop_update_start_loop: int = 2,
        loop_update_schedule_slope: float = 0.0,
        loop_update_norm_eps: float = 1.0e-6,
        loop_input_dropout: float = 0.0,
        loop_input_dropout_start: int = 2,
        loop_noise_std: float = 0.0,
        loop_noise_mode: str = "relative",
        loop_noise_start_loop: int = 1,
        loop_noise_warmup_steps: int = 0,
        loop_noise_after_last_loop: bool = False,
        **kwargs: Any,
    ) -> None:
        self.num_loops = num_loops
        self.loop_update_mode = loop_update_mode
        self.loop_update_alpha = loop_update_alpha
        self.loop_update_start_loop = loop_update_start_loop
        self.loop_update_schedule_slope = loop_update_schedule_slope
        self.loop_update_norm_eps = loop_update_norm_eps
        self.loop_input_dropout = loop_input_dropout
        self.loop_input_dropout_start = loop_input_dropout_start
        self.loop_noise_std = loop_noise_std
        self.loop_noise_mode = loop_noise_mode
        self.loop_noise_start_loop = loop_noise_start_loop
        self.loop_noise_warmup_steps = loop_noise_warmup_steps
        self.loop_noise_after_last_loop = loop_noise_after_last_loop
        super().__init__(**kwargs)
        if self.num_loops < 1:
            raise ValueError("num_loops must be positive")
        valid_update_modes = {
            "full",
            "fixed",
            "learned_shared",
            "normalized_projected",
            "normalized_projected_learned",
        }
        if self.loop_update_mode not in valid_update_modes:
            raise ValueError(
                "loop_update_mode must be one of " + ", ".join(sorted(valid_update_modes))
            )
        if self.loop_update_start_loop < 1:
            raise ValueError("loop_update_start_loop must be positive")
        if self.loop_update_norm_eps <= 0.0:
            raise ValueError("loop_update_norm_eps must be positive")
        if self.loop_update_mode.startswith("normalized_projected"):
            if self.loop_update_start_loop < 2:
                raise ValueError("normalized projected updates must start at loop 2 or later")
            if not 0.0 < self.loop_update_alpha < 1.0:
                raise ValueError("normalized projected loop_update_alpha must be in (0, 1)")
        if not 0.0 <= self.loop_input_dropout < 1.0:
            raise ValueError("loop_input_dropout must be in [0, 1)")
        if self.loop_input_dropout_start < 1:
            raise ValueError("loop_input_dropout_start must be positive")
        if self.loop_noise_std < 0.0:
            raise ValueError("loop_noise_std must be non-negative")
        if self.loop_noise_mode not in {"relative", "norm_preserving"}:
            raise ValueError("loop_noise_mode must be relative or norm_preserving")
        if self.loop_noise_start_loop < 1:
            raise ValueError("loop_noise_start_loop must be positive")
        if self.loop_noise_warmup_steps < 0:
            raise ValueError("loop_noise_warmup_steps must be non-negative")
        if self.use_cache:
            # A standard per-layer Qwen3 KV cache is not loop-aware.
            self.use_cache = False


class LoopedQwen3Model(Qwen3Model):
    """Official Qwen3Model with its layer stack recurrently reused."""

    config_class = LoopedQwen3Config

    def __init__(self, config: LoopedQwen3Config) -> None:
        super().__init__(config)
        if config.loop_update_mode == "learned_shared":
            self.loop_update_alpha = torch.nn.Parameter(
                torch.tensor(float(config.loop_update_alpha))
            )
        else:
            self.register_buffer(
                "loop_update_alpha",
                torch.tensor(float(config.loop_update_alpha)),
                persistent=False,
            )
        if config.loop_update_mode == "normalized_projected_learned":
            initial_alpha = float(config.loop_update_alpha)
            initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
            self.loop_update_logit = torch.nn.Parameter(torch.tensor(initial_logit))
            self.loop_update_log_slope = torch.nn.Parameter(
                torch.tensor(float(config.loop_update_schedule_slope))
            )
        self.register_buffer("_loop_noise_multiplier", torch.ones(()), persistent=False)
        self.last_relative_updates: tuple[torch.Tensor, ...] = ()
        self.last_hidden_norms: tuple[torch.Tensor, ...] = ()
        self.last_cosine_to_previous: tuple[torch.Tensor, ...] = ()
        self.last_loop_update_alphas: tuple[torch.Tensor, ...] = ()
        self.last_captured_hidden_states: tuple[torch.Tensor, ...] = ()
        self.last_captured_loops: tuple[int, ...] = ()

    def set_loop_noise_step(self, step: int) -> float:
        """Set the noise warmup multiplier for an optimizer step."""
        warmup = self.config.loop_noise_warmup_steps
        multiplier = 1.0 if warmup == 0 else min((step + 1) / warmup, 1.0)
        self._loop_noise_multiplier.fill_(multiplier)
        return multiplier

    def current_loop_update_alpha(self, loop_idx: int | None = None) -> torch.Tensor:
        if self.config.loop_update_mode == "full":
            return self.embed_tokens.weight.new_ones(())
        if self.config.loop_update_mode in {"fixed", "normalized_projected"}:
            # Non-persistent buffers may be left uninitialized by Transformers'
            # low-memory from_pretrained path. Fixed gates are configuration,
            # not learned state, so reconstruct them from the serialized config.
            return self.embed_tokens.weight.new_tensor(float(self.config.loop_update_alpha))
        if self.config.loop_update_mode == "normalized_projected_learned":
            if loop_idx is None:
                loop_idx = self.config.num_loops - 1
            loop_number = loop_idx + 1
            relative_loop = max(loop_number / self.config.loop_update_start_loop, 1.0)
            schedule_position = self.loop_update_logit.new_tensor(math.log(relative_loop))
            return torch.sigmoid(
                self.loop_update_logit + self.loop_update_log_slope * schedule_position
            )
        return self.loop_update_alpha

    def _normalized_projected_update(
        self,
        previous: torch.Tensor,
        proposal: torch.Tensor,
        reference_rms: torch.Tensor,
        loop_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Take a normalized step and return each token to the reference RMS sphere."""
        original_dtype = proposal.dtype
        previous_float = previous.float()
        delta = proposal.float() - previous_float
        eps = self.config.loop_update_norm_eps
        delta_rms = delta.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        normalized_delta = delta * (reference_rms / delta_rms)
        alpha = self.current_loop_update_alpha(loop_idx).float()
        candidate = previous_float + alpha * normalized_delta
        candidate_rms = candidate.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        projected = candidate * (reference_rms / candidate_rms)
        return projected.to(original_dtype), alpha

    def _add_loop_noise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add scale-relative noise; optionally keep each token on its RMS sphere."""
        original_dtype = hidden_states.dtype
        states = hidden_states.float()
        signal_rms = states.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        noise = torch.randn_like(states)

        if self.config.loop_noise_mode == "norm_preserving":
            # Remove the radial component. The perturbation explores a tangent
            # direction and the final projection prevents recurrent norm drift.
            projection = (noise * states).mean(dim=-1, keepdim=True)
            projection = projection / signal_rms.square()
            noise = noise - projection * states

        noise = noise / noise.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        std = self.config.loop_noise_std * self._loop_noise_multiplier
        perturbed = states + std * signal_rms * noise

        if self.config.loop_noise_mode == "norm_preserving":
            perturbed_rms = perturbed.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            perturbed = perturbed * (signal_rms / perturbed_rms)
        return perturbed.to(original_dtype)

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
        capture_hidden_at_loops: tuple[int, ...] | list[int] | None = None,
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
        capture_loops = tuple(capture_hidden_at_loops or ())
        if capture_loops != tuple(sorted(set(capture_loops))):
            raise ValueError("capture_hidden_at_loops must be sorted and unique")
        if any(loop < 1 or loop > loops for loop in capture_loops):
            raise ValueError("captured loop depths must be between 1 and num_loops")
        capture_set = set(capture_loops)
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
        hidden_norms = []
        cosine_to_previous = []
        loop_update_alphas = []
        captured_hidden_states = []
        reference_rms = None

        for loop_idx in range(loops):
            previous = hidden_states
            if (
                self.training
                and self.config.loop_input_dropout > 0.0
                and loop_idx + 1 >= self.config.loop_input_dropout_start
            ):
                hidden_states = torch.nn.functional.dropout(
                    hidden_states,
                    p=self.config.loop_input_dropout,
                    training=True,
                )
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

            if self.config.loop_update_mode.startswith("normalized_projected"):
                loop_number = loop_idx + 1
                if loop_number < self.config.loop_update_start_loop:
                    alpha = hidden_states.new_ones(())
                    if loop_number == self.config.loop_update_start_loop - 1:
                        reference_rms = (
                            hidden_states.float()
                            .square()
                            .mean(dim=-1, keepdim=True)
                            .sqrt()
                            .clamp_min(self.config.loop_update_norm_eps)
                            .detach()
                        )
                else:
                    if reference_rms is None:
                        raise RuntimeError("Missing reference RMS for normalized loop update")
                    hidden_states, alpha = self._normalized_projected_update(
                        previous, hidden_states, reference_rms, loop_idx
                    )
            else:
                alpha = self.current_loop_update_alpha(loop_idx).to(hidden_states.dtype)
                hidden_states = previous + alpha * (hidden_states - previous)

            if return_loop_diagnostics:
                numerator = (hidden_states.float() - previous.float()).norm(dim=-1).mean()
                denominator = previous.float().norm(dim=-1).mean().clamp_min(1e-8)
                relative_updates.append((numerator / denominator).detach())
                hidden_norms.append(hidden_states.float().norm(dim=-1).mean().detach())
                cosine = torch.nn.functional.cosine_similarity(
                    hidden_states.float(), previous.float(), dim=-1
                ).mean()
                cosine_to_previous.append(cosine.detach())
                loop_update_alphas.append(alpha.detach().float())

            # Capture before optional between-loop noise. This is exactly the
            # state that would be decoded if computation stopped at this depth.
            if loop_idx + 1 in capture_set:
                captured_hidden_states.append(hidden_states)

            should_add_noise = (
                self.training
                and self.config.loop_noise_std > 0.0
                and loop_idx + 1 >= self.config.loop_noise_start_loop
                and (loop_idx + 1 < loops or self.config.loop_noise_after_last_loop)
            )
            if should_add_noise:
                hidden_states = self._add_loop_noise(hidden_states)

        self.last_relative_updates = tuple(relative_updates)
        self.last_hidden_norms = tuple(hidden_norms)
        self.last_cosine_to_previous = tuple(cosine_to_previous)
        self.last_loop_update_alphas = tuple(loop_update_alphas)
        self.last_captured_hidden_states = tuple(captured_hidden_states)
        self.last_captured_loops = capture_loops
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

    def set_loop_noise_step(self, step: int) -> float:
        return self.model.set_loop_noise_step(step)

    @staticmethod
    def _causal_token_losses(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shifted per-token CE and its valid-token mask."""
        shift_logits = logits[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        valid = shift_labels.ne(-100)
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        return token_losses, valid

    @staticmethod
    def _hard_token_weights(
        previous_token_losses: torch.Tensor,
        valid: torch.Tensor,
        gamma: float,
        minimum: float,
        maximum: float,
    ) -> torch.Tensor:
        """Build detached, per-sequence-normalized weights from an earlier exit."""
        valid_float = valid.to(previous_token_losses.dtype)
        per_sequence_mean = (
            (previous_token_losses.detach() * valid_float).sum(dim=-1, keepdim=True)
            / valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ).clamp_min(1.0e-8)
        relative_difficulty = previous_token_losses.detach() / per_sequence_mean
        weights = relative_difficulty.clamp_min(0.0).pow(gamma).clamp(minimum, maximum)
        return torch.where(valid, weights, torch.zeros_like(weights))

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
        supervision_loops: tuple[int, ...] | list[int] | None = None,
        supervision_weights: tuple[float, ...] | list[float] | None = None,
        token_loss_weighting: str = "uniform",
        hard_token_gamma: float = 0.5,
        hard_token_min_weight: float = 0.25,
        hard_token_max_weight: float = 4.0,
        hard_token_uniform_mix: float = 0.5,
        **kwargs: Any,
    ) -> LoopedCausalLMOutputWithPast:
        # This mirrors the thin official Qwen3ForCausalLM wrapper. Custom loop
        # arguments are consumed here instead of leaking into the HF loss.
        loops = self.config.num_loops if num_loops is None else num_loops
        supervised_loops = tuple(supervision_loops or ())
        if supervised_loops and labels is None:
            raise ValueError("labels are required when supervision_loops are provided")
        if supervised_loops != tuple(sorted(set(supervised_loops))):
            raise ValueError("supervision_loops must be sorted and unique")
        if any(loop < 1 or loop > loops for loop in supervised_loops):
            raise ValueError("supervision depths must be between 1 and num_loops")
        if token_loss_weighting not in {"uniform", "previous_loss"}:
            raise ValueError("token_loss_weighting must be uniform or previous_loss")
        if hard_token_gamma <= 0.0:
            raise ValueError("hard_token_gamma must be positive")
        if not 0.0 < hard_token_min_weight <= hard_token_max_weight:
            raise ValueError("hard-token clipping must satisfy 0 < min <= max")
        if not 0.0 <= hard_token_uniform_mix <= 1.0:
            raise ValueError("hard_token_uniform_mix must be in [0, 1]")
        if token_loss_weighting == "previous_loss" and (
            not isinstance(logits_to_keep, int) or logits_to_keep != 0
        ):
            raise ValueError("previous-loss token weighting requires full-sequence logits")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            num_loops=num_loops,
            return_loop_diagnostics=return_loop_diagnostics,
            capture_hidden_at_loops=[loop for loop in supervised_loops if loop != loops],
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        loop_losses = None
        loop_uniform_losses = None
        loop_corrective_losses = None
        loop_hard_weight_means = None
        if labels is not None and supervised_loops:
            if supervision_weights is None:
                weights = logits.new_full((len(supervised_loops),), 1.0 / len(supervised_loops))
            else:
                if len(supervision_weights) != len(supervised_loops):
                    raise ValueError("supervision_weights must match supervision_loops")
                weights = logits.new_tensor(supervision_weights)
                if torch.any(weights < 0) or weights.sum() <= 0:
                    raise ValueError("supervision_weights must be non-negative with a positive sum")
                weights = weights / weights.sum()

            captured = dict(
                zip(self.model.last_captured_loops, self.model.last_captured_hidden_states)
            )
            depth_logits_by_loop = {}
            for loop in supervised_loops:
                if loop == loops:
                    depth_logits = logits
                else:
                    depth_hidden = self.model.norm(captured[loop])
                    depth_logits = self.lm_head(depth_hidden[:, slice_indices, :])
                depth_logits_by_loop[loop] = depth_logits

            losses = []
            uniform_losses = []
            corrective_losses = []
            hard_weight_means = []
            previous_token_losses = None
            for loop in supervised_loops:
                token_losses, valid = self._causal_token_losses(
                    depth_logits_by_loop[loop], labels
                )
                valid_float = valid.to(token_losses.dtype)
                uniform_loss = (token_losses * valid_float).sum() / valid_float.sum().clamp_min(1.0)
                corrective_loss = uniform_loss
                mean_hard_weight = uniform_loss.new_ones(())
                if token_loss_weighting == "previous_loss" and previous_token_losses is not None:
                    hard_weights = self._hard_token_weights(
                        previous_token_losses,
                        valid,
                        gamma=hard_token_gamma,
                        minimum=hard_token_min_weight,
                        maximum=hard_token_max_weight,
                    )
                    corrective_loss = (token_losses * hard_weights).sum() / hard_weights.sum().clamp_min(1.0)
                    mean_hard_weight = hard_weights.sum() / valid_float.sum().clamp_min(1.0)
                mixed_loss = (
                    hard_token_uniform_mix * uniform_loss
                    + (1.0 - hard_token_uniform_mix) * corrective_loss
                )
                losses.append(mixed_loss)
                uniform_losses.append(uniform_loss)
                corrective_losses.append(corrective_loss)
                hard_weight_means.append(mean_hard_weight)
                previous_token_losses = token_losses
            loop_losses = torch.stack(losses)
            loop_uniform_losses = torch.stack(uniform_losses)
            loop_corrective_losses = torch.stack(corrective_losses)
            loop_hard_weight_means = torch.stack(hard_weight_means)
            loss = (loop_losses * weights).sum()
        elif labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return LoopedCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            loop_losses=loop_losses,
            loop_uniform_losses=loop_uniform_losses,
            loop_corrective_losses=loop_corrective_losses,
            loop_hard_weight_means=loop_hard_weight_means,
            supervised_loops=supervised_loops or None,
        )


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
