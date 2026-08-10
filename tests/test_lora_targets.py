from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sonomed_vlm.config import LoraConfigModel
from sonomed_vlm.models.lora import apply_lora, find_decoder_lora_targets, parameter_report


def _tiny_model() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    )


def test_decoder_targets_exist_and_one_backward_step_changes_lora() -> None:
    model = _tiny_model()
    config = LoraConfigModel(rank=2, alpha=4, dropout=0.0)
    targets = find_decoder_lora_targets(model, config.target_modules)
    assert targets
    adapted, selected = apply_lora(model, config)
    assert selected == targets
    report = parameter_report(adapted)
    assert 0 < report.trainable < report.total
    # A one-layer 16-wide fixture has unusually high adapter overhead; the real
    # 4B model should be far lower, but this still catches accidental full tuning.
    assert report.trainable_percent < 20
    base_parameters = [
        parameter for name, parameter in adapted.named_parameters() if "lora_" not in name
    ]
    assert base_parameters and all(not parameter.requires_grad for parameter in base_parameters)
    trainable = [
        (name, parameter)
        for name, parameter in adapted.named_parameters()
        if parameter.requires_grad
    ]
    before = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=0.1)
    input_ids = torch.randint(0, 64, (2, 8))
    loss = adapted(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0) for _, parameter in trainable
    )
    optimizer.step()
    assert any(not torch.equal(before[name], parameter) for name, parameter in trainable)


def test_missing_target_fails_loudly() -> None:
    try:
        find_decoder_lora_targets(_tiny_model(), ["not_a_real_projection"])
    except ValueError as exc:
        assert "do not exist" in str(exc)
    else:
        raise AssertionError("missing target should fail")
