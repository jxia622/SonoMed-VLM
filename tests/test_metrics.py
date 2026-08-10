from __future__ import annotations

import pytest

from sonomed_vlm.eval.classification import mcq_score, parse_multiple_choice
from sonomed_vlm.eval.generation import exact_match, rouge_l_f1, token_f1
from sonomed_vlm.eval.grounding import box_iou, parse_box
from sonomed_vlm.eval.parsing import score_prediction


@pytest.mark.parametrize(
    ("output", "expected"),
    [("B", "B"), ("B: Kidney", "B"), ("The answer is B.", "B"), ("Kidney", "B")],
)
def test_multiple_choice_parser(output: str, expected: str) -> None:
    parsed = parse_multiple_choice(output, ["A: Liver", "B: Kidney"])
    assert parsed.valid and parsed.label == expected


def test_contradictory_multiple_choice_scores_label_and_text_separately() -> None:
    score = mcq_score(
        "A: Thyroid",
        "D",
        ["A: Fetus", "B: Mammary", "C: Skin", "D: Thyroid"],
    )
    assert score["strict_label_accuracy"] == 0.0
    assert score["option_text_accuracy"] == 1.0
    assert score["semantic_choice_accuracy"] == 1.0
    assert score["parsed_semantic_choice"] == "D"
    assert score["label_text_consistency"] == 0.0
    assert score["contradiction_rate"] == 1.0
    assert score["valid_response_rate"] == 1.0


def test_option_text_normalization_removes_harmless_formatting() -> None:
    score = mcq_score("** Thyroid. **", "D", ["A: Fetus", "D: Thyroid"])
    assert score["strict_label_accuracy"] == 0.0
    assert score["option_text_accuracy"] == 1.0
    assert score["semantic_choice_accuracy"] == 1.0
    assert score["contradiction_rate"] == 0.0


@pytest.mark.parametrize(
    "output",
    ["B", "The answer is B", "Kidney", "B: Kidney", "A: Kidney"],
)
def test_semantic_choice_resolution_is_format_independent(output: str) -> None:
    score = mcq_score(output, "B", ["A: Liver", "B: Kidney", "C: Thyroid", "D: Spleen"])
    assert score["semantic_choice_accuracy"] == 1.0
    assert score["parsed_semantic_choice"] == "B"


def test_bare_option_text_starting_with_label_character_is_not_a_label() -> None:
    score = mcq_score("Breast", "D", ["A: Liver", "B: Kidney", "C: Thyroid", "D: Breast"])
    assert score["parsed_label"] is None
    assert score["parsed_option_text_label"] == "D"
    assert score["semantic_choice_accuracy"] == 1.0


@pytest.mark.parametrize("output", ["B", "The answer is B", "Kidney", "A: Kidney"])
def test_semantic_choice_supports_unlabeled_ordered_options(output: str) -> None:
    score = mcq_score(output, "B", ["Liver", "Kidney", "Thyroid", "Spleen"])
    assert score["parsed_semantic_choice"] == "B"
    assert score["semantic_choice_accuracy"] == 1.0


def test_semantic_choice_finds_unique_option_text_in_explanation_before_label() -> None:
    score = mcq_score(
        "Based on the image, the organ is **Kidney**. Therefore, the answer is A.",
        "B",
        ["A: Liver", "B: Kidney", "C: Thyroid", "D: Spleen"],
    )
    assert score["parsed_semantic_option_text_label"] == "B"
    assert score["parsed_semantic_label"] == "A"
    assert score["parsed_semantic_choice"] == "B"
    assert score["semantic_choice_accuracy"] == 1.0


def test_semantic_choice_falls_back_to_explicit_answer_label() -> None:
    score = mcq_score(
        "The anatomy is unclear. Therefore, the correct answer is **B**.",
        "B",
        ["A: Liver", "B: Kidney", "C: Thyroid", "D: Spleen"],
    )
    assert score["parsed_semantic_option_text_label"] is None
    assert score["parsed_semantic_label"] == "B"
    assert score["semantic_choice_accuracy"] == 1.0


def test_semantic_choice_prefers_longest_specific_option_text() -> None:
    score = mcq_score("This is a simple cyst.", "B", ["A: Cyst", "B: Simple cyst"])
    assert score["parsed_semantic_choice"] == "B"
    assert score["semantic_choice_accuracy"] == 1.0


def test_generation_metrics_known_values() -> None:
    assert exact_match("The kidney.", "the kidney") == 1.0
    assert token_f1("kidney normal", "kidney") == pytest.approx(2 / 3)
    assert rouge_l_f1("normal left kidney", "left kidney") == pytest.approx(0.8)


def test_grounding_parser_and_iou() -> None:
    box = parse_box("[0.1, 0.2, 0.5, 0.8]")
    assert box.valid and box.coordinates is not None
    assert box_iou(box.coordinates, box.coordinates) == 1.0
    assert not parse_box("nan, 0, 1, 1").valid
    assert not parse_box("0.5, 0.5, 0.2, 0.9").valid


def test_grounding_score_accepts_sonoinstruct_coordinates() -> None:
    score = score_prediction(
        {
            "raw_model_output": "```json\n[321, 216, 487, 510]\n```",
            "ground_truth": "```json\n[321, 216, 487, 510]\n```",
            "task_type": "detection",
            "task_family": "VG",
        }
    )
    assert score["valid_box_rate"] == 1.0
    assert score["invalid_box_rate"] == 0.0
    assert score["iou"] == 1.0
    assert score["localization_at_0_5"] == 1.0


def test_grounding_score_rejects_coordinates_above_sonoinstruct_range() -> None:
    score = score_prediction(
        {
            "raw_model_output": "[0, 0, 1001, 500]",
            "ground_truth": "[0, 0, 1000, 500]",
            "task_type": "detection",
            "task_family": "VG",
        }
    )
    assert score["valid_box_rate"] == 0.0
    assert score["invalid_box_rate"] == 1.0
