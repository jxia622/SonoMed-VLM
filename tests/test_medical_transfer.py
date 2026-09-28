import importlib.util
import sys
from pathlib import Path

import pytest

from sonomed_vlm.config import ExperimentConfig
from sonomed_vlm.data.collator import build_messages
from sonomed_vlm.data.text_qa import TextQADataset
from sonomed_vlm.models.lora import apply_lora, load_trainable_adapter, parameter_report
from sonomed_vlm.utils.io import sha256_json, write_jsonl

spec = importlib.util.spec_from_file_location(
    "prepare_medical_transfer", Path(__file__).parents[1] / "scripts/prepare_medical_transfer.py"
)
prep = importlib.util.module_from_spec(spec)
# Scripts bootstrap their source tree; allow its local import during tests.

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec.loader.exec_module(prep)


def test_token_counting_handles_transformers_mapping_return():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {
                "input_ids": [1, 2, 3] if kwargs.get("add_generation_prompt") else [1, 2, 3, 4, 5]
            }

    r = prep.record("Question?", "Answer", "test", "train", "0", Tokenizer())
    assert (r["prompt_tokens"], r["answer_tokens"], r["total_tokens"]) == (3, 2, 5)


def test_overlap_detects_reformatted_and_near_duplicate_vignettes():
    index = prep.OverlapIndex()
    q = "A forty year old person presents with persistent pain in the right upper abdomen after meals for several weeks and a low grade fever. What is the most likely diagnosis?"
    index.add(q)
    assert index.matches(q.upper().replace(" ", "  "))
    assert index.matches(q.replace("forty", "forty-five"))
    assert not index.matches("What planet is closest to the Sun?")


def test_no_options_conversion_rejects_unanswerable_questions():
    assert (
        prep.med_question("Which of the following is the most likely diagnosis?", "Pneumonia")
        == "What is the most likely diagnosis?"
    )
    for q, a in [
        ("Which of the following is incorrect?", "X"),
        ("What does the image show?", "Pneumonia"),
        ("Which diagnosis is likely?", "All of the above"),
    ]:
        with pytest.raises(ValueError):
            prep.med_question(q, a)


def test_matched_pairs_have_identical_supervised_token_counts():
    med = [{"example_id": f"m{i}", "answer_tokens": 7, "total_tokens": 200} for i in range(16)]
    general = [{"example_id": f"g{i}", "answer_tokens": 7, "total_tokens": 202} for i in range(16)]
    general += [{"example_id": "bad", "answer_tokens": 8, "total_tokens": 200}]
    pairs = prep.match_pairs(med, general)
    assert len(pairs) == 16
    assert len({g["example_id"] for m, g in pairs}) == 16
    assert all(m["answer_tokens"] == g["answer_tokens"] for m, g in pairs)


def test_text_dataset_does_not_insert_dummy_images_or_candidate_answers(tmp_path):
    r = {
        "example_id": "1",
        "source": "test",
        "question": "What organ filters blood?",
        "answer": "Kidney",
    }
    r["content_sha256"] = sha256_json({"q": r["question"], "a": r["answer"]})
    file = tmp_path / "data.jsonl"
    write_jsonl(file, [r])
    example = TextQADataset(file)[0]
    messages = build_messages(example)
    assert example.images == []
    assert messages[1]["content"] == [{"type": "text", "text": r["question"]}]
    assert messages[2]["content"][0]["text"] == "Kidney"
    r["answer"] = "Changed"
    write_jsonl(file, [r])
    with pytest.raises(ValueError, match="digest"):
        TextQADataset(file)


def test_adapter_transition_preserves_weights_and_capacity(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = ExperimentConfig.model_validate(
        {
            "model": {"name": "tiny-test"},
            "data": {"root": str(tmp_path)},
            "output": {"root": str(tmp_path)},
        }
    )
    tiny = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    tiny._name_or_path = "tiny-test"
    original, _ = apply_lora(LlamaForCausalLM(tiny), cfg.lora)
    for _n, p in original.named_parameters():
        if p.requires_grad:
            with torch.no_grad():
                p.fill_(0.123)
    original.save_pretrained(tmp_path / "adapter")
    restored, _ = load_trainable_adapter(LlamaForCausalLM(tiny), cfg, tmp_path / "adapter")
    assert parameter_report(restored).trainable == parameter_report(original).trainable
    for n, p in restored.named_parameters():
        if p.requires_grad:
            assert "lora_" in n
            assert torch.allclose(p, torch.full_like(p, 0.123))
    cfg.lora.rank = 8
    with pytest.raises(ValueError, match="recipe"):
        load_trainable_adapter(LlamaForCausalLM(tiny), cfg, tmp_path / "adapter")


def test_submission_dag_gates_all_runs_and_avoids_expired_dependencies(tmp_path, monkeypatch):
    import csv

    import submit_medical_transfer as submit

    calls = []

    def fake_command(command, **kwargs):
        if command[0] == "sacct":
            return "COMPLETED\n"
        calls.append(command)
        return f"{50000+len(calls)};gpu\n"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SONOMED_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["submit_medical_transfer.py"])
    monkeypatch.setattr(submit.subprocess, "check_output", fake_command)
    submit.main()
    assert len(calls) == 36
    assert not any(arg.startswith("--dependency") for arg in calls[0])
    assert all(any(arg.startswith("--dependency=afterok:") for arg in cmd) for cmd in calls[1:])
    assert not any("4079527" in arg for cmd in calls for arg in cmd)
    summary_dependency = next(arg for arg in calls[-1] if arg.startswith("--dependency"))
    assert len(summary_dependency.split(":")) == 36  # afterok + all 35 prior jobs
    receipt = list(csv.DictReader((tmp_path / "submission-medical-transfer.tsv").open(), delimiter="\t"))
    assert len(receipt) == 36
    assert all(row["cluster"] == "gpu" for row in receipt)
    with pytest.raises(FileExistsError):
        submit.main()
