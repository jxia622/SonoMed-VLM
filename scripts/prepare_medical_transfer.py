#!/usr/bin/env python3
"""Freeze and audit a medical-vs-general intermediate QA experiment on CRC."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from sonomed_vlm.data.collator import _input_ids
from sonomed_vlm.data.open_qa import DEPENDENT_ANSWER, DEPENDENT_QUESTION
from sonomed_vlm.data.text_qa import SYSTEM
from sonomed_vlm.utils.io import read_jsonl, sha256_file, sha256_json, write_json, write_jsonl

SOURCES = {
    "medqa": (
        "GBaker/MedQA-USMLE-4-options",
        "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
        ["phrases_no_exclude_train.jsonl", "phrases_no_exclude_test.jsonl"],
    ),
    "squad": (
        "rajpurkar/squad",
        "7b6d24c440a36b6815f21b70d25016731768db1f",
        ["plain_text/train-00000-of-00001.parquet", "plain_text/validation-00000-of-00001.parquet"],
    ),
}
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MEDICAL = re.compile(
    r"\b(?:medical|medicine|clinical|patient|disease|diagnosis|diagnostic|hospital|physician|cancer|surgery|surgical|anatomy|ultrasound|pharmacology|drug|drugs|symptom|symptoms|therapy|therapeutic)\b",
    re.I,
)


def norm(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def shingles(text):
    words = norm(text).split()
    return {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


class OverlapIndex:
    """Exact normalized questions + lexical near duplicates; not a semantic guarantee.

    Candidate retrieval uses up to 20 rarest shared word 5-grams, ignoring
    boilerplate present in >1000 indexed questions. Match requires >=80%
    5-gram containment in either direction and >=10 shared 5-grams.
    """

    def __init__(self):
        self.exact = set()
        self.grams = []
        self.postings = defaultdict(list)

    def add(self, text):
        key = norm(text)
        if key in self.exact:
            return
        self.exact.add(key)
        gs = shingles(text)
        idx = len(self.grams)
        self.grams.append(gs)
        for g in gs:
            self.postings[g].append(idx)

    def matches(self, text):
        if norm(text) in self.exact:
            return True
        gs = shingles(text)
        candidates = set()
        rare = sorted(
            (g for g in gs if 0 < len(self.postings.get(g, ())) <= 1000),
            key=lambda g: (len(self.postings[g]), g),
        )[:20]
        for g in rare:
            candidates.update(self.postings[g])
        for i in candidates:
            other = self.grams[i]
            common = len(gs & other)
            if common >= 10 and common / min(len(gs), len(other)) >= 0.8:
                return True
        return False


def med_question(question, answer):
    # Mechanical grammar-only rewrite; reject questions that still require choices.
    question = re.sub(r"\bwhich of the following\b", "What", question, flags=re.I)
    question = re.sub(r"\bwhat of the following\b", "What", question, flags=re.I)
    if (
        DEPENDENT_QUESTION.search(question)
        or DEPENDENT_ANSWER.search(answer)
        or re.search(
            r"\b(?:except|least likely|not true|incorrect|false statement|following|shown|pictured|figure|image|diagram|exhibit)\b",
            question,
            re.I,
        )
        or re.search(r"\bwhat\s+(?:is|are)\s+(?:true|correct|false)\b", question, re.I)
        or re.fullmatch(r"[A-E]", answer.strip(), re.I)
    ):
        raise ValueError("choice_or_image_dependent")
    return question.strip()


def record(q, a, source, split, source_id, tokenizer):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    prompt = _input_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    )
    full = _input_ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": a}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full[: len(prompt)] != prompt or len(full) > 2048:
        raise ValueError("length_or_template")
    return {
        "example_id": sha256_json({"source": source, "split": split, "id": source_id}),
        "source": source,
        "source_split": split,
        "source_id": source_id,
        "question": q,
        "answer": a,
        "content_sha256": sha256_json({"q": q, "a": a}),
        "prompt_tokens": len(prompt),
        "answer_tokens": len(full) - len(prompt),
        "total_tokens": len(full),
    }


def match_pairs(med, general):
    """Match unique examples within 5% total length and exact answer-token count."""
    bins = defaultdict(list)
    for r in general:
        bins[r["answer_tokens"]].append(r)
    rng = random.Random(20260928)
    for rows in bins.values():
        rng.shuffle(rows)
    used = set()
    pairs = []
    for m in sorted(med, key=lambda r: (-r["answer_tokens"], r["example_id"])):
        best = None
        for alen in [m["answer_tokens"]]:
            for g in bins.get(alen, []):
                if g["example_id"] in used:
                    continue
                delta = abs(g["total_tokens"] - m["total_tokens"])
                if delta <= max(2, 0.05 * m["total_tokens"]):
                    score = (abs(alen - m["answer_tokens"]) * 10000 + delta, g["example_id"])
                    if best is None or score < best[0]:
                        best = (score, g)
        if best:
            used.add(best[1]["example_id"])
            pairs.append((m, best[1]))
    rng.shuffle(pairs)
    # Exact divisible batch count: no DDP padding and identical optimizer updates.
    return pairs[: len(pairs) // 8 * 8]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--sono-manifests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    audit = {
        "sources": {},
        "filters": {},
        "overlap_method": OverlapIndex.__doc__,
        "model": MODEL,
        "model_revision": REVISION,
    }
    source_rows = {}
    for name, (repo, revision, files) in SOURCES.items():
        paths = [
            Path(hf_hub_download(repo, f, repo_type="dataset", revision=revision)) for f in files
        ]
        audit["sources"][name] = {
            "repo": repo,
            "revision": revision,
            "files": {f: sha256_file(path) for f, path in zip(files, paths, strict=True)},
        }
        for split, path in zip(["train", "test"], paths, strict=True):
            source_rows[name, split] = (
                list(read_jsonl(path))
                if path.suffix == ".jsonl"
                else pq.read_table(path).to_pylist()
            )
    # Audit against ALL released SonoInstruct questions, not only this split.
    sono = OverlapIndex()
    source_counts = Counter()
    for shard in sorted((args.data_root / "parquet_full_train").glob("*.parquet")):
        for row in pq.read_table(shard, columns=["QA", "dataset"]).to_pylist():
            items = json.loads(row["QA"]) if isinstance(row["QA"], str) else row["QA"]
            source_counts[str(row["dataset"])] += len(items)
            for qa in items:
                q = re.sub(r"\s*<image>\s*", "\n", str(qa.get("question") or ""), flags=re.I)
                sono.add(q)
                # Same canonicalization used for candidate clinical questions.
                sono.add(re.sub(r"\bwhich of the following\b", "What", q, flags=re.I))
        print("overlap indexed", shard.name, len(sono.exact), flush=True)
    audit["sono_source_counts"] = dict(source_counts)
    pools = {}
    # Held-outs are processed first so all new train examples can be checked against them.
    heldout = OverlapIndex()
    for split in ["test", "train"]:
        for source in ["medqa", "squad"]:
            kept = []
            reasons = Counter()
            within = OverlapIndex()
            for i, row in enumerate(source_rows[source, split]):
                try:
                    if source == "medqa":
                        a = row["answer"].strip()
                        if row["options"][row["answer_idx"]].strip() != a:
                            raise ValueError("gold_disagreement")
                        q = med_question(row["question"], a)
                        check_text = q
                    else:
                        if MEDICAL.search(row["context"] + " " + row["question"]):
                            raise ValueError("medical_general_control")
                        a = row["answers"]["text"][0].strip()
                        q = (
                            "Passage:\n"
                            + row["context"].strip()
                            + "\n\nQuestion: "
                            + row["question"].strip()
                        )
                        check_text = row["question"]
                    if sono.matches(check_text):
                        raise ValueError("sono_overlap")
                    if within.matches(check_text):
                        raise ValueError("duplicate_within_source")
                    if split == "train" and (heldout.matches(check_text) or heldout.matches(q)):
                        raise ValueError("heldout_overlap")
                    r = record(q, a, source, split, str(row.get("id", i)), tokenizer)
                    kept.append(r)
                    within.add(check_text)
                    if split == "test":
                        heldout.add(check_text)
                        heldout.add(q)
                except ValueError as exc:
                    reasons[str(exc)] += 1
            pools[source, split] = kept
            audit["filters"][source + "_" + split] = {
                "original": len(source_rows[source, split]),
                "kept": len(kept),
                "excluded": dict(reasons),
            }
            print(source, split, audit["filters"][source + "_" + split], flush=True)
    pairs = match_pairs(pools["medqa", "train"], pools["squad", "train"])
    if len(pairs) < 1000:
        raise ValueError(
            f"Only {len(pairs)} uncontaminated matched pairs; redesign instead of silently weakening controls"
        )
    dev_pairs, pairs = pairs[-128:], pairs[:-128]
    chosen = {
        "medical_train": [m for m, g in pairs],
        "general_train": [g for m, g in pairs],
        "medical_dev": [m for m, g in dev_pairs],
        "general_dev": [g for m, g in dev_pairs],
    }
    for source, label in [("medqa", "medical_test"), ("squad", "general_test")]:
        rows = sorted(pools[source, "test"], key=lambda r: r["example_id"])
        if len(rows) < 200:
            raise ValueError(f"Insufficient clean held-out examples: {source} {len(rows)}")
        chosen[label] = rows[:500]
    stats = {}
    for name, rows in chosen.items():
        stats[name] = {
            "examples": len(rows),
            **{
                key: sum(r[key] for r in rows)
                for key in ["prompt_tokens", "answer_tokens", "total_tokens"]
            },
        }
    for key in ["prompt_tokens", "answer_tokens", "total_tokens"]:
        ratio = stats["general_train"][key] / stats["medical_train"][key]
        if abs(ratio - 1) > 0.02:
            raise ValueError(f"Aggregate {key} not matched within 2%: {ratio}")
    for name, rows in chosen.items():
        write_jsonl(args.output / (name + ".jsonl"), rows)
        stats[name]["sha256"] = sha256_file(args.output / (name + ".jsonl"))
    # Separate smoke artifacts: never become experimental initial checkpoints.
    for arm in ["medical", "general"]:
        write_jsonl(args.output / (arm + "_smoke_train.jsonl"), chosen[arm + "_train"][:32])
        write_jsonl(args.output / (arm + "_smoke_eval.jsonl"), chosen[arm + "_dev"][:16])
    val = list(read_jsonl(args.sono_manifests / "val.jsonl"))
    by_task = defaultdict(list)
    for r in val:
        by_task[str(r["task_type"])].append(r)
    smoke = [r for rows in by_task.values() for r in rows[:8]]
    write_jsonl(args.output / "ultrasound_smoke_eval.jsonl", smoke)
    audit["manifests"] = stats
    audit["sono_manifests"] = {
        f: sha256_file(args.sono_manifests / f)
        for f in ["train_1pct.jsonl", "train_10pct.jsonl", "train.jsonl", "val.jsonl"]
    }
    audit["limitations"] = [
        "Lexical overlap screening does not guarantee semantic or translation deduplication.",
        "Public benchmark exposure during Qwen pretraining is unknown.",
        "SQuAD is passage-based general QA; reasoning demand differs from clinical vignettes.",
        "Open-ended lexical scores are not clinician-adjudicated correctness.",
    ]
    write_json(args.output / "audit.json", audit)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
