from dataclasses import replace

import pytest

from sonomed_vlm.data.collator import build_messages
from sonomed_vlm.data.open_qa import PROTOCOL, ConversionError, apply_protocol, canonical_answer
from sonomed_vlm.data.schema import ImageAsset, NormalizedExample
from sonomed_vlm.eval.parsing import score_prediction


def example(png_bytes):
    return NormalizedExample(
        "id",
        "image",
        [ImageAsset(image_bytes=png_bytes)],
        "source",
        "AR",
        "mcq",
        "Kidney",
        "Only output the option letter.",
        "Which organ is shown?",
        "B: Kidney",
        {"answer_label": "B", "options": ["A: Liver", "B: Kidney"]},
    )


def test_conversion_removes_options_and_labels_from_both_model_turns(png_bytes):
    original = example(png_bytes)
    converted = apply_protocol(original, PROTOCOL)
    messages = build_messages(converted)
    assert converted.example_id == original.example_id
    assert converted.task_type == "open_qa"
    assert converted.assistant_response == "Kidney"
    assert converted.metadata["options"] == [] and converted.metadata["answer_label"] is None
    assert messages[1]["content"][-1]["text"] == original.user_prompt
    assert messages[-1]["content"][0]["text"] == "Kidney"
    assert "Only output the option letter." not in str(messages)
    assert "Liver" not in str(messages)
    assert original.assistant_response == "B: Kidney"  # Historical recipe remains intact.
    assert apply_protocol(original, "legacy") is original


def test_non_mcq_tasks_are_unchanged(png_bytes):
    original = replace(
        example(png_bytes), task_type="detection", assistant_response="[0, 0, 100, 100]"
    )
    converted = apply_protocol(original, PROTOCOL)
    assert converted.task_type == original.task_type
    assert converted.system_prompt == original.system_prompt
    assert converted.assistant_response == original.assistant_response


@pytest.mark.parametrize(
    "raw,expected",
    [("Kidney", 1), ("A: Kidney", 1), ("B", 0), ("The answer is B", 0), ("Liver", 0), ("", 0)],
)
def test_scoring_cannot_recover_an_answer_from_an_option_letter(raw, expected):
    score = score_prediction(
        {
            "task_type": "open_qa",
            "raw_model_output": raw,
            "ground_truth": "Kidney",
            "answer_label": "B",
            "options": ["A: Liver", "B: Kidney"],
        }
    )
    assert score["answer_exact_match"] == expected
    assert "semantic_choice_accuracy" not in score
    assert "strict_label_accuracy" not in score


@pytest.mark.parametrize("answer", ["A kidney", "B-cell lymphoma", "T2 hyperintensity", "12 mm"])
def test_preserves_clinical_letters_and_measurements(answer):
    result = score_prediction(
        {"task_type": "open_qa", "raw_model_output": answer, "ground_truth": answer}
    )
    assert result["parsed_prediction"] == answer
    assert result["answer_exact_match"] == 1


@pytest.mark.parametrize(
    "answer,label,options,question",
    [
        ("A: Kidney", "B", ["A: Liver", "B: Kidney"], "Which organ?"),
        ("B: Liver", "B", ["A: Liver", "B: Kidney"], "Which organ?"),
        ("B: All of the above", "B", ["A: Liver", "B: All of the above"], "Which organ?"),
        ("B: Kidney", "B", ["A: Liver", "B: Kidney"], "Which of the following is correct?"),
        ("B: Kidney", "B", ["A: Liver", "B: Kidney"], "Which organ?\nA: Liver\nB: Kidney"),
    ],
)
def test_ambiguous_or_inconsistent_source_annotations_fail_closed(answer, label, options, question):
    with pytest.raises(ConversionError):
        canonical_answer(answer, label, options, question)


def test_gold_source_label_can_be_resolved_but_not_a_prediction_label():
    assert canonical_answer("B", "B", ["A: Liver", "B: Kidney"], "Which organ?") == "Kidney"


@pytest.mark.parametrize(
    "answer,label,options,expected",
    [
        ("10 × 10 mm", "B", ["6 × 8 mm", "10 × 10 mm"], "10 × 10 mm"),  # noqa: RUF001
        (
            "A directly over the calcification",
            "A",
            ["A directly over the calcification", "B deep to the calcification"],
            "directly over the calcification",
        ),
        (
            "A. Umbilical cord around the neck",
            "A",
            ["A. Umbilical cord around the neck", "B. Normal"],
            "Umbilical cord around the neck",
        ),
    ],
)
def test_real_source_formats_preserve_measurements_and_clinical_words(
    answer, label, options, expected
):
    assert canonical_answer(answer, label, options, "What is observed?") == expected


@pytest.mark.parametrize("prediction,reference", [("P > 0.05", "P < 0.05"), ("25 mm", "2.5 mm")])
def test_exact_matching_preserves_comparators_and_decimal_points(prediction, reference):
    result = score_prediction(
        {"task_type": "open_qa", "raw_model_output": prediction, "ground_truth": reference}
    )
    assert result["answer_exact_match"] == 0


def test_choice_bearing_records_mislabeled_as_qa_are_also_corrected(png_bytes):
    original = replace(example(png_bytes), task_type="qa")
    converted = apply_protocol(original, PROTOCOL)
    assert converted.task_type == "open_qa"
    assert converted.assistant_response == "Kidney"
    assert converted.metadata["original_task_type"] == "qa"


def test_ordinary_qa_remains_qa(png_bytes):
    original = replace(example(png_bytes), task_type="qa", metadata={}, assistant_response="Kidney")
    assert apply_protocol(original, PROTOCOL).task_type == "qa"
