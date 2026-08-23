import torch
import torch.nn.functional as F

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
