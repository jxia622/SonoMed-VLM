from __future__ import annotations

from sonomed_vlm.data.splits import (
    assert_no_image_leakage,
    attach_leakage_groups,
    nested_subsets,
    split_records,
)


def _record(index: int, image_ids: list[str]) -> dict:
    return {
        "example_id": f"e{index}",
        "image_id": image_ids[0],
        "image_ids": image_ids,
        "source_dataset": f"source-{index % 3}",
        "task_family": f"task-{index % 2}",
    }


def test_overlapping_multi_image_rows_form_one_component() -> None:
    records = [_record(0, ["a", "b"]), _record(1, ["b", "c"]), _record(2, ["d"])]
    grouped = attach_leakage_groups(records)
    assert grouped[0]["leakage_group_id"] == grouped[1]["leakage_group_id"]
    assert grouped[2]["leakage_group_id"] != grouped[0]["leakage_group_id"]


def test_split_has_no_image_leakage_and_is_deterministic() -> None:
    records = [_record(index, [f"image-{index}"]) for index in range(60)]
    first = split_records(records, val_fraction=0.2, seed=7)
    second = split_records(records, val_fraction=0.2, seed=7)
    assert [row["example_id"] for row in first[0]] == [row["example_id"] for row in second[0]]
    assert_no_image_leakage(*first)
    assert first[0] and first[1]


def test_scaling_subsets_are_nested_by_group() -> None:
    records = [_record(index, [f"image-{index}"]) for index in range(100)]
    subsets = nested_subsets(records, seed=42)
    previous: set[str] = set()
    for fraction in (1, 5, 10, 25, 50, 100):
        current = {row["example_id"] for row in subsets[fraction]}
        assert previous <= current
        previous = current
    assert len(subsets[100]) == 100
