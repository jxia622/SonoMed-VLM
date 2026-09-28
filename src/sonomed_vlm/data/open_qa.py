"""Versioned conversion of self-contained MCQs into answer-text-only QA."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from sonomed_vlm.data.schema import NormalizedExample

PROTOCOL = "open_qa_v2"
SYSTEM_PROMPT = "Answer the ultrasound question with the answer text only. Be concise. Do not output an option letter or a list of choices."
MCQ_TYPES = {"mcq", "multiple_choice", "multiple-choice"}
# Require punctuation so clinical text such as 'A kidney', 'B-cell lymphoma',
# 'T2 hyperintensity', and numeric measurements is not silently shortened.
PREFIX = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])\s*[:.)])\s*", re.I)
LABEL_ONLY = re.compile(
    r"^\s*(?:\(?[A-Z]\)?[.:]?|(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)\s*\(?[A-Z]\)?[.:]?)\s*$",
    re.I,
)
DEPENDENT_ANSWER = re.compile(
    r"\b(?:all|none|both|neither|any)\s+of\s+(?:the\s+)?(?:above|below|following|options|choices)\b|"
    r"\b(?:all|none)\s+(?:are\s+)?(?:correct|incorrect|true|false)\b|"
    r"\b(?:both\s+)?[A-D](?:\s+(?:and|or)\s+|\s*&\s*)[A-D]\b|"
    r"\b(?:option|choice|statement)s?\s+[A-D]\b",
    re.I,
)
DEPENDENT_QUESTION = re.compile(
    r"\b(?:which|what)\s+of\s+(?:the\s+)?following\b|"
    r"\b(?:following|above|below|listed)\s+(?:options?|choices?|statements?|answers?)\b|"
    r"\b(?:option|choice)s?\s+[A-D]\b|"
    r"\b(?:choose|select)\s+(?:the\s+)?(?:correct\s+)?(?:option|choice)\b|"
    r"(?:^|\n)\s*\(?[A-D][.):]\s",
    re.I,
)


class ConversionError(ValueError):
    """An example cannot be converted without changing or guessing its meaning."""


def is_choice_example(task_type: str | None, metadata: dict[str, Any]) -> bool:
    kind = str(task_type or "").casefold()
    if "ground" in kind or kind in {"detection", "vg"}:
        return False
    return kind in MCQ_TYPES or bool(metadata.get("options") and metadata.get("answer_label"))


def strip_explicit_label(text: str) -> str:
    return PREFIX.sub("", text, count=1).strip()


def canonical_answer(answer: str, answer_label: Any, options: Any, question: str) -> str:
    if not isinstance(options, list) or not options or not isinstance(answer_label, str):
        raise ConversionError("missing_choice_metadata")
    if any(not isinstance(option, str) for option in options) or len(options) > 26:
        raise ConversionError("non_text_or_excess_options")
    matches = [PREFIX.match(option) for option in options]
    space_matches = [re.match(rf"^\s*{chr(65 + i)}\s+", option) for i, option in enumerate(options)]
    explicit = all(matches)
    spaced = not any(matches) and all(space_matches)
    if any(matches) and not explicit:
        raise ConversionError("mixed_option_label_formats")
    mapping = {}
    for index, option in enumerate(options):
        match = matches[index] if explicit else space_matches[index] if spaced else None
        label = (match.group(1) or match.group(2)).upper() if explicit else chr(65 + index)
        if label in mapping:
            raise ConversionError("duplicate_source_label")
        mapping[label] = option[match.end() :].strip() if match else option.strip()
    gold = mapping.get(answer_label.strip().upper())
    if not gold:
        raise ConversionError("missing_gold_option")
    match = PREFIX.match(answer)
    if match and (match.group(1) or match.group(2)).upper() != answer_label.strip().upper():
        raise ConversionError("source_label_conflict")
    # Labels may be resolved for source annotations, never for model predictions.
    if not LABEL_ONLY.fullmatch(answer):

        def norm(value: str) -> str:
            return " ".join(value.casefold().split())

        source_text = strip_explicit_label(answer)
        if spaced:
            source_text = re.sub(
                rf"^\s*{re.escape(answer_label.strip().upper())}\s+", "", source_text
            )
        if norm(source_text) != norm(gold):
            raise ConversionError("source_answer_text_conflict")
    if DEPENDENT_ANSWER.search(gold) or LABEL_ONLY.fullmatch(gold):
        raise ConversionError("choice_dependent_answer")
    if re.match(r"^\s*based on\b", question, re.I) and not re.search(
        r"\b(?:what|which|where|how|why|when|who|is|are|does|identify|describe|classify|determine|diagnose|interpret|provide|assess|name)\b",
        question,
        re.I,
    ):
        raise ConversionError("incomplete_question")
    if DEPENDENT_QUESTION.search(question):
        raise ConversionError("choice_dependent_question")
    return gold


def apply_protocol(example: NormalizedExample, protocol: str) -> NormalizedExample:
    if protocol == "legacy":
        return example
    if protocol != PROTOCOL:
        raise ValueError(f"Unknown instruction protocol: {protocol}")
    metadata = {
        **example.metadata,
        "instruction_protocol": protocol,
        "original_task_type": example.task_type,
    }
    if not is_choice_example(example.task_type, metadata):
        return replace(example, metadata=metadata)
    gold = canonical_answer(
        example.assistant_response,
        metadata.get("answer_label"),
        metadata.get("options"),
        example.user_prompt,
    )
    metadata.update({"answer_label": None, "options": [], "choice": None})
    return replace(
        example,
        task_type="open_qa",
        system_prompt=SYSTEM_PROMPT,
        assistant_response=gold,
        metadata=metadata,
    )
