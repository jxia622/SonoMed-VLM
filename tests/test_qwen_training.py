from __future__ import annotations

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from sonomed_vlm.config import LoraConfigModel
from sonomed_vlm.models.lora import apply_lora


def test_qwen_image_forward_backward_updates_only_decoder_lora() -> None:
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=60, video_token_id=61,
        vision_start_token_id=62, vision_end_token_id=63,
    )
    model, targets = apply_lora(
        Qwen3VLForConditionalGeneration(config), LoraConfigModel(rank=2, alpha=4)
    )
    assert len(targets) == 7
    assert all("visual" not in name for name in targets)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    ids = torch.tensor([[1, 62, 60, 60, 60, 60, 63, 2, 3]])
    labels = ids.clone()
    labels[:, :7] = -100
    loss = model(
        input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(16, 3 * 2 * 16 * 16),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        mm_token_type_ids=ids.eq(60).long(),
    ).loss
    assert torch.isfinite(loss)
    loss.backward()
    torch.optim.AdamW(trainable, lr=0.01).step()
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable, strict=True))
    assert all(not p.requires_grad for name, p in model.named_parameters() if "visual" in name)
