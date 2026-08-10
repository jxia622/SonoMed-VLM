from __future__ import annotations

from collections import UserDict
from typing import Any

import torch

from sonomed_vlm.data.collator import AssistantOnlyMultimodalCollator, _input_ids, build_messages
from sonomed_vlm.data.schema import ImageAsset, NormalizedExample


class FakeProcessor:
    def _encode(self, conversation: list[dict[str, Any]], add_generation_prompt: bool) -> list[int]:
        tokens = [1]
        for message in conversation:
            role = message["role"]
            if role == "system":
                tokens.extend([10, 11])
            elif role == "user":
                tokens.append(20)
                for item in message["content"]:
                    if item["type"] == "image":
                        tokens.append(21)
                    else:
                        tokens.extend([30] * max(1, len(item["text"].split())))
            elif role == "assistant":
                tokens.append(40)
                tokens.extend([50] * max(1, len(message["content"][0]["text"].split())))
                tokens.append(2)
        if add_generation_prompt:
            tokens.append(40)
        return tokens

    def apply_chat_template(self, conversations, add_generation_prompt=False, **kwargs):
        is_batch = bool(conversations) and isinstance(conversations[0], list)
        values = conversations if is_batch else [conversations]
        encoded = [self._encode(value, add_generation_prompt) for value in values]
        if kwargs.get("return_tensors") == "pt":
            width = max(map(len, encoded))
            ids = torch.zeros((len(encoded), width), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, value in enumerate(encoded):
                ids[index, width - len(value) :] = torch.tensor(value)
                mask[index, width - len(value) :] = 1
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "pixel_values": torch.zeros((len(encoded), 1, 3, 2, 2)),
            }
        return {"input_ids": encoded if is_batch else encoded[0]}


def _example(image: bytes, answer: str) -> NormalizedExample:
    return NormalizedExample(
        example_id=answer,
        image_id="id",
        images=[ImageAsset(image_bytes=image)],
        source_dataset="source",
        task_family="AR",
        task_type="qa",
        focus="Kidney",
        system_prompt="Be concise.",
        user_prompt="What is shown?",
        assistant_response=answer,
    )


def test_messages_contain_real_images_and_official_roles(png_bytes: bytes) -> None:
    messages = build_messages(_example(png_bytes, "Kidney"))
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][0]["image"].size == (8, 6)


def test_input_ids_accepts_mapping_objects() -> None:
    feature = UserDict({"input_ids": [[1, 2, 3]]})
    assert _input_ids(feature) == [1, 2, 3]


def test_assistant_only_and_padding_loss_mask(png_bytes: bytes) -> None:
    processor = FakeProcessor()
    examples = [_example(png_bytes, "Kidney"), _example(png_bytes, "Normal left kidney")]
    batch = AssistantOnlyMultimodalCollator(processor)(examples)
    for row, example in enumerate(examples):
        prompt = processor.apply_chat_template(
            build_messages(example, include_answer=False),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )["input_ids"]
        active = batch["attention_mask"][row].bool()
        active_positions = active.nonzero(as_tuple=False).flatten()
        assert torch.all(batch["labels"][row, active_positions[: len(prompt)]] == -100)
        assert torch.all(batch["labels"][row, ~active] == -100)
        assert torch.any(batch["labels"][row, active_positions[len(prompt) :]] != -100)
    assert batch["pixel_values"].shape == (2, 1, 3, 2, 2)
