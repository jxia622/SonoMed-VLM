from __future__ import annotations

from sonomed_vlm.data.schema import ImageAsset, NormalizedExample
from sonomed_vlm.data.validation import ValidationReport, validate_example


def test_valid_and_invalid_examples_are_counted(png_bytes: bytes) -> None:
    valid = NormalizedExample(
        example_id="valid",
        image_id="image",
        images=[ImageAsset(image_bytes=png_bytes)],
        source_dataset=None,
        task_family="AR",
        task_type="qa",
        focus=None,
        system_prompt=None,
        user_prompt="Question",
        assistant_response="Answer",
    )
    invalid = NormalizedExample(
        example_id="invalid",
        image_id="image",
        images=[ImageAsset(image_bytes=b"not-an-image")],
        source_dataset=None,
        task_family="AR",
        task_type="qa",
        focus=None,
        system_prompt=None,
        user_prompt="",
        assistant_response="",
    )
    report = ValidationReport()
    assert validate_example(valid, report)
    assert not validate_example(invalid, report)
    assert report.valid == 1
    assert report.exclusions["empty_question"] == 1
    assert report.exclusions["empty_response"] == 1
    assert report.exclusions["unreadable_image"] == 1
