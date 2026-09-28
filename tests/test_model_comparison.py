from __future__ import annotations

from copy import deepcopy

import pytest

from sonomed_vlm.eval.comparison import compare_predictions


def rows():
    settings = {"do_sample": False, "max_new_tokens": 256}
    return [
        {"example_id": "mcq", "image_id": "one", "task_type": "mcq",
         "prompt": "Which organ?", "ground_truth": "A: Kidney", "answer_label": "A",
         "options": ["A: Kidney", "B: Liver"], "raw_model_output": "B: Liver",
         "generation_settings": settings},
        {"example_id": "box", "image_id": "two", "task_family": "VG",
         "prompt": "Locate it", "ground_truth": "[100, 100, 500, 500]",
         "raw_model_output": "invalid", "generation_settings": settings},
    ]


def test_comparison_rescores_and_aligns_by_id_including_invalid_outputs():
    base = rows()
    tuned = deepcopy(base)
    tuned[0]["raw_model_output"] = "A: Kidney"
    tuned[0]["score"] = {"strict_label_accuracy": 0.0}  # stale scores must be ignored
    tuned[1]["raw_model_output"] = "[100, 100, 500, 500]"
    report = compare_predictions(
        {"qwen_original": base, "qwen_finetuned": list(reversed(tuned))}, ["mcq", "box"]
    )
    assert report["models"]["qwen_original"]["overall"]["iou"] == 0.0
    assert report["models"]["qwen_finetuned"]["overall"]["strict_label_accuracy"] == 1.0
    assert report["comparisons"][0]["metrics"]["iou"]["difference"] == 1.0


@pytest.mark.parametrize("field,value", [("prompt", "different"), ("answer_label", "B"),
                                         ("generation_settings", {"do_sample": True})])
def test_comparison_rejects_unmatched_protocol(field, value):
    changed = deepcopy(rows())
    changed[0][field] = value
    with pytest.raises(ValueError):
        compare_predictions({"qwen_original": rows(), "qwen_finetuned": changed}, ["mcq", "box"])


def test_comparison_rejects_duplicate_or_missing_examples():
    for invalid in (rows()[:1], [rows()[0], rows()[0]]):
        with pytest.raises(ValueError):
            compare_predictions({"qwen_original": invalid}, ["mcq", "box"])
