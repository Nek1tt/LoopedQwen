import torch
import torch.nn.functional as F
from types import MethodType

from looped_lm import LoopedQwen3Config, LoopedQwen3ForCausalLM


def tiny_config(**overrides):
    values = dict(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_loops=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        use_cache=False,
        tie_word_embeddings=True,
    )
    values.update(overrides)
    return LoopedQwen3Config(**values)


def test_forward_backward():
    model = LoopedQwen3ForCausalLM(tiny_config())
    x = torch.randint(0, 257, (2, 16))
    output = model(input_ids=x, labels=x, return_loop_diagnostics=True)
    assert output.logits.shape == (2, 16, 257)
    assert len(model.model.last_relative_updates) == 3
    output.loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_loop_override_changes_result():
    torch.manual_seed(0)
    model = LoopedQwen3ForCausalLM(tiny_config()).eval()
    x = torch.randint(0, 257, (1, 8))
    with torch.no_grad():
        one = model(input_ids=x, num_loops=1).logits
        three = model(input_ids=x, num_loops=3).logits
    assert not torch.allclose(one, three)


def test_tied_embeddings_are_counted_once():
    model = LoopedQwen3ForCausalLM(tiny_config(tie_word_embeddings=True))
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_hugging_face_roundtrip(tmp_path):
    model = LoopedQwen3ForCausalLM(tiny_config())
    model.save_pretrained(tmp_path)
    restored = LoopedQwen3ForCausalLM.from_pretrained(tmp_path)
    assert restored.config.num_loops == 3
    assert len(restored.model.layers) == 2


def test_fixed_update_alpha_survives_hugging_face_roundtrip(tmp_path):
    model = LoopedQwen3ForCausalLM(
        tiny_config(
            loop_update_mode="normalized_projected",
            loop_update_alpha=0.25,
            loop_update_start_loop=2,
        )
    )
    model.save_pretrained(tmp_path)
    restored = LoopedQwen3ForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(
        restored.model.current_loop_update_alpha(1),
        torch.tensor(0.25),
    )


def test_same_physical_layers_are_reused_each_loop():
    model = LoopedQwen3ForCausalLM(tiny_config(num_loops=3)).eval()
    calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.model.layers[0].register_forward_hook(count_call)
    with torch.no_grad():
        model(input_ids=torch.randint(0, 257, (1, 8)))
    handle.remove()
    assert calls == 3
    assert len(model.model.layers) == 2


def test_hf_causal_loss_shifts_labels_once():
    model = LoopedQwen3ForCausalLM(tiny_config()).eval()
    x = torch.randint(0, 257, (2, 8))
    with torch.no_grad():
        output = model(input_ids=x, labels=x)
        expected = F.cross_entropy(
            output.logits[:, :-1].contiguous().view(-1, 257),
            x[:, 1:].contiguous().view(-1),
        )
    torch.testing.assert_close(output.loss, expected)


def test_fixed_loop_update_is_applied():
    torch.manual_seed(0)
    full = LoopedQwen3ForCausalLM(tiny_config(num_loops=1)).eval()
    fixed = LoopedQwen3ForCausalLM(
        tiny_config(num_loops=1, loop_update_mode="fixed", loop_update_alpha=0.0)
    ).eval()
    fixed.load_state_dict(full.state_dict(), strict=False)
    x = torch.randint(0, 257, (1, 8))
    with torch.no_grad():
        fixed_hidden = fixed.model(input_ids=x).last_hidden_state
        embedded = fixed.model.norm(fixed.model.embed_tokens(x))
    torch.testing.assert_close(fixed_hidden, embedded)


def test_loop_input_dropout_is_training_only():
    model = LoopedQwen3ForCausalLM(
        tiny_config(loop_input_dropout=0.2, loop_input_dropout_start=2)
    )
    x = torch.randint(0, 257, (1, 8))
    model.eval()
    with torch.no_grad():
        first = model(input_ids=x).logits
        second = model(input_ids=x).logits
    torch.testing.assert_close(first, second)


def test_norm_preserving_loop_noise_preserves_token_rms():
    config = tiny_config(loop_noise_std=0.1, loop_noise_mode="norm_preserving")
    model = LoopedQwen3ForCausalLM(config)
    states = torch.randn(2, 5, config.hidden_size)
    before = states.float().square().mean(dim=-1).sqrt()
    after = model.model._add_loop_noise(states)
    after_rms = after.float().square().mean(dim=-1).sqrt()
    torch.testing.assert_close(after_rms, before, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(after, states)


def test_loop_noise_warmup_and_eval_determinism():
    model = LoopedQwen3ForCausalLM(
        tiny_config(loop_noise_std=0.05, loop_noise_warmup_steps=10)
    )
    assert model.set_loop_noise_step(0) == 0.1
    assert model.set_loop_noise_step(20) == 1.0
    x = torch.randint(0, 257, (1, 8))
    model.eval()
    with torch.no_grad():
        first = model(input_ids=x).logits
        second = model(input_ids=x).logits
    torch.testing.assert_close(first, second)


def test_normalized_projected_update_preserves_reference_rms():
    config = tiny_config(
        loop_update_mode="normalized_projected",
        loop_update_alpha=0.25,
        loop_update_start_loop=2,
    )
    model = LoopedQwen3ForCausalLM(config)
    previous = torch.randn(2, 5, config.hidden_size)
    proposal = previous + 3.0 * torch.randn_like(previous)
    reference_rms = previous.float().square().mean(dim=-1, keepdim=True).sqrt()
    updated, alpha = model.model._normalized_projected_update(
        previous, proposal, reference_rms, loop_idx=1
    )
    updated_rms = updated.float().square().mean(dim=-1, keepdim=True).sqrt()
    torch.testing.assert_close(updated_rms, reference_rms, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alpha, torch.tensor(0.25))


def test_projected_forward_locks_norm_after_first_loop():
    model = LoopedQwen3ForCausalLM(
        tiny_config(
            num_loops=4,
            loop_update_mode="normalized_projected",
            loop_update_alpha=0.25,
            loop_update_start_loop=2,
        )
    ).eval()
    with torch.no_grad():
        model(
            input_ids=torch.randint(0, 257, (2, 8)),
            return_loop_diagnostics=True,
        )
    norms = torch.stack(model.model.last_hidden_norms)
    torch.testing.assert_close(norms[1:], norms[0].expand_as(norms[1:]), rtol=1e-5, atol=1e-6)
    assert [value.item() for value in model.model.last_loop_update_alphas] == [
        1.0,
        0.25,
        0.25,
        0.25,
    ]


def test_learned_projected_schedule_has_gradients_and_extrapolates():
    model = LoopedQwen3ForCausalLM(
        tiny_config(
            num_loops=4,
            loop_update_mode="normalized_projected_learned",
            loop_update_alpha=0.25,
            loop_update_start_loop=2,
            loop_update_schedule_slope=0.0,
        )
    )
    initial = [model.model.current_loop_update_alpha(i).item() for i in (1, 3, 7)]
    assert initial == [0.25, 0.25, 0.25]
    model.model.loop_update_log_slope.data.fill_(0.5)
    extrapolated = [model.model.current_loop_update_alpha(i).item() for i in (1, 3, 7)]
    assert extrapolated[0] < extrapolated[1] < extrapolated[2]

    output = model(input_ids=torch.randint(0, 257, (2, 8)), labels=torch.randint(0, 257, (2, 8)))
    output.loss.backward()
    assert model.model.loop_update_logit.grad is not None
    assert model.model.loop_update_log_slope.grad is not None


def test_intermediate_lm_losses_match_independent_depths():
    torch.manual_seed(0)
    model = LoopedQwen3ForCausalLM(tiny_config(num_loops=4)).eval()
    x = torch.randint(0, 257, (2, 8))
    with torch.no_grad():
        anytime = model(
            input_ids=x,
            labels=x,
            num_loops=4,
            supervision_loops=(1, 3, 4),
            supervision_weights=(0.25, 0.25, 0.5),
        )
        independent = torch.stack(
            [model(input_ids=x, labels=x, num_loops=depth).loss for depth in (1, 3, 4)]
        )
    torch.testing.assert_close(anytime.loop_losses, independent)
    torch.testing.assert_close(
        anytime.loss,
        (independent * independent.new_tensor([0.25, 0.25, 0.5])).sum(),
    )
    assert anytime.supervised_loops == (1, 3, 4)


def test_intermediate_losses_backpropagate_to_shared_layers():
    model = LoopedQwen3ForCausalLM(tiny_config(num_loops=4))
    x = torch.randint(0, 257, (2, 8))
    output = model(
        input_ids=x,
        labels=x,
        num_loops=4,
        supervision_loops=(1, 2, 4),
        supervision_weights=(0.25, 0.25, 0.5),
    )
    output.loss.backward()
    assert output.loop_losses.shape == (3,)
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_invalid_supervision_plan_is_rejected():
    model = LoopedQwen3ForCausalLM(tiny_config(num_loops=4))
    x = torch.randint(0, 257, (1, 8))
    with torch.no_grad():
        try:
            model(input_ids=x, labels=x, num_loops=4, supervision_loops=(3, 2, 4))
        except ValueError as error:
            assert "sorted and unique" in str(error)
        else:
            raise AssertionError("unsorted supervision depths were accepted")


def test_hard_token_weights_are_detached_and_prioritize_difficult_tokens():
    previous = torch.tensor([[1.0, 4.0, 0.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    weights = LoopedQwen3ForCausalLM._hard_token_weights(
        previous,
        valid,
        gamma=0.5,
        minimum=0.25,
        maximum=4.0,
    )
    assert not weights.requires_grad
    assert weights[0, 1] > weights[0, 0]
    assert weights[0, 2] == 0.0


def test_previous_loss_weighting_matches_manual_corrective_objective():
    torch.manual_seed(0)
    model = LoopedQwen3ForCausalLM(tiny_config(num_loops=4)).eval()
    x = torch.randint(0, 257, (2, 8))
    with torch.no_grad():
        output = model(
            input_ids=x,
            labels=x,
            num_loops=4,
            supervision_loops=(1, 3, 4),
            supervision_weights=(0.25, 0.25, 0.5),
            token_loss_weighting="previous_loss",
            hard_token_gamma=0.5,
            hard_token_min_weight=0.25,
            hard_token_max_weight=4.0,
            hard_token_uniform_mix=0.5,
        )
        independent_logits = [
            model(input_ids=x, num_loops=depth).logits for depth in (1, 3, 4)
        ]

    expected_mixed = []
    previous_token_losses = None
    for depth_logits in independent_logits:
        token_losses, valid = model._causal_token_losses(depth_logits, x)
        valid_float = valid.to(token_losses.dtype)
        uniform = (token_losses * valid_float).sum() / valid_float.sum()
        corrective = uniform
        if previous_token_losses is not None:
            hard_weights = model._hard_token_weights(
                previous_token_losses,
                valid,
                gamma=0.5,
                minimum=0.25,
                maximum=4.0,
            )
            corrective = (token_losses * hard_weights).sum() / hard_weights.sum()
        expected_mixed.append(0.5 * uniform + 0.5 * corrective)
        previous_token_losses = token_losses

    expected_mixed = torch.stack(expected_mixed)
    torch.testing.assert_close(output.loop_losses, expected_mixed)
    torch.testing.assert_close(
        output.loss,
        (expected_mixed * expected_mixed.new_tensor([0.25, 0.25, 0.5])).sum(),
    )


def test_invalid_hard_token_configuration_is_rejected():
    model = LoopedQwen3ForCausalLM(tiny_config(num_loops=2))
    x = torch.randint(0, 257, (1, 8))
    with torch.no_grad():
        try:
            model(
                input_ids=x,
                labels=x,
                supervision_loops=(1, 2),
                token_loss_weighting="previous_loss",
                hard_token_min_weight=2.0,
                hard_token_max_weight=1.0,
            )
        except ValueError as error:
            assert "min <= max" in str(error)
        else:
            raise AssertionError("invalid hard-token clipping was accepted")


def test_attention_mlp_schedule_matches_official_qwen_layer_forward():
    torch.manual_seed(0)
    model = LoopedQwen3ForCausalLM(
        tiny_config(num_loops=3, loop_operator_schedule="attention_mlp")
    ).eval()
    reference = LoopedQwen3ForCausalLM(model.config).eval()
    reference.load_state_dict(model.state_dict())

    def official_apply(
        self,
        decoder_layer,
        hidden_states,
        attention_mask,
        position_embeddings,
        position_ids,
        order,
        **kwargs,
    ):
        assert order == "attention_mlp"
        output = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        )
        zero = torch.zeros_like(output)
        return output, zero, zero

    reference.model._apply_decoder_layer = MethodType(official_apply, reference.model)
    x = torch.randint(0, 257, (2, 8))
    with torch.no_grad():
        actual = model(input_ids=x).logits
        expected = reference(input_ids=x).logits
    torch.testing.assert_close(actual, expected)


def test_alternating_schedule_calls_sublayers_in_expected_order():
    model = LoopedQwen3ForCausalLM(
        tiny_config(num_loops=3, loop_operator_schedule="alternating_attention_mlp")
    ).eval()
    calls = []
    layer = model.model.layers[0]
    attention_handle = layer.self_attn.register_forward_pre_hook(
        lambda _module, _inputs: calls.append("attention")
    )
    mlp_handle = layer.mlp.register_forward_pre_hook(
        lambda _module, _inputs: calls.append("mlp")
    )
    with torch.no_grad():
        model(input_ids=torch.randint(0, 257, (1, 8)))
    attention_handle.remove()
    mlp_handle.remove()
    assert calls == [
        "attention",
        "mlp",
        "mlp",
        "attention",
        "attention",
        "mlp",
    ]
    assert model.model.last_operator_orders == (
        "attention_mlp",
        "mlp_attention",
        "attention_mlp",
    )


def test_operator_schedules_have_identical_parameter_counts_but_different_outputs():
    torch.manual_seed(0)
    fixed_am = LoopedQwen3ForCausalLM(
        tiny_config(num_loops=4, loop_operator_schedule="attention_mlp")
    ).eval()
    alternating = LoopedQwen3ForCausalLM(
        tiny_config(num_loops=4, loop_operator_schedule="alternating_attention_mlp")
    ).eval()
    alternating.load_state_dict(fixed_am.state_dict())
    assert sum(p.numel() for p in fixed_am.parameters()) == sum(
        p.numel() for p in alternating.parameters()
    )
    x = torch.randint(0, 257, (1, 8))
    with torch.no_grad():
        am_logits = fixed_am(input_ids=x).logits
        alternating_logits = alternating(input_ids=x).logits
    assert not torch.allclose(am_logits, alternating_logits)


def test_operator_schedule_override_is_causal_and_does_not_mutate_config():
    torch.manual_seed(0)
    model = LoopedQwen3ForCausalLM(
        tiny_config(num_loops=4, loop_operator_schedule="alternating_attention_mlp")
    ).eval()
    x = torch.randint(0, 257, (1, 8))
    with torch.no_grad():
        native = model(input_ids=x).logits
        forced_am = model(
            input_ids=x, loop_operator_schedule="attention_mlp"
        ).logits
        reversed_phase = model(
            input_ids=x, loop_operator_schedule="alternating_mlp_attention"
        ).logits
    assert not torch.allclose(native, forced_am)
    assert not torch.allclose(native, reversed_phase)
    assert model.config.loop_operator_schedule == "alternating_attention_mlp"


def test_operator_diagnostics_are_finite_and_nonzero():
    model = LoopedQwen3ForCausalLM(
        tiny_config(num_loops=4, loop_operator_schedule="alternating_attention_mlp")
    ).eval()
    with torch.no_grad():
        model(
            input_ids=torch.randint(0, 257, (2, 8)),
            return_loop_diagnostics=True,
            return_operator_diagnostics=True,
        )
    diagnostic_groups = (
        model.model.last_attention_relative_updates,
        model.model.last_mlp_relative_updates,
        model.model.last_operator_defects,
        model.model.last_update_cosine_to_previous_update,
        model.model.last_directional_diversity,
    )
    assert all(len(values) == 4 for values in diagnostic_groups)
    assert all(torch.isfinite(torch.stack(values)).all() for values in diagnostic_groups)
    assert torch.stack(model.model.last_operator_defects).min() > 0


def test_operator_schedule_survives_hugging_face_roundtrip(tmp_path):
    model = LoopedQwen3ForCausalLM(
        tiny_config(loop_operator_schedule="alternating_attention_mlp")
    )
    model.save_pretrained(tmp_path)
    restored = LoopedQwen3ForCausalLM.from_pretrained(tmp_path)
    assert restored.config.loop_operator_schedule == "alternating_attention_mlp"
