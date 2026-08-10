# Data pipeline

## Observed public schema

The public `Ssdaizi/SonoInstruct` release was inspected on 2026-08-07 through Hugging Face repository metadata and the Parquet footers of all full-training shards, without downloading the 15.8 GB payload.

- 53 shards under `parquet_full_train/`.
- 263,247 source rows.
- 52 shards contain 5,000 rows; the final shard contains 3,247.
- Total compressed size observed: 15,812,192,112 bytes.
- All shards have the same fields; 20 place `info` last and 33 place it before `type`/`images`.

| Column | Observed Arrow type | Meaning |
|---|---|---|
| `QA` | `large_string` | JSON-encoded list of QA objects |
| `dataset` | `large_string` | source dataset |
| `focus` | `large_string` | anatomy/topic focus |
| `info` | `large_string` | JSON-encoded source metadata |
| `type` | `large_string` | row-level task/family label |
| `images` | `list<struct<bytes: binary>>` | one or more embedded original images |

Observed full-release QA objects contain `answer`, `answer_label`, `options`, `qa_type`, `question`, and `system_prompt`. The preview is different: it has `image_path`, omits `info`, and presents `QA` as an Arrow list of structs with a QA-level `type`. The adapter supports both formats. Runtime inspection remains mandatory because the upstream repository can change.

Run:

```bash
python scripts/inspect_dataset.py --data-root "$SONOINSTRUCT_ROOT"
```

It writes `artifacts/dataset_schema.json` and `artifacts/dataset_schema.md`, including every shard, Arrow types, row counts, safely truncated examples, image/QA counts, sampled image modes/dimensions, corruption counts, and sampled duplicate hashes. Embedded bytes are never serialized.

## Normalization and flattening

Each source row is converted into one internal example per QA item. All QA items retain the same ordered image identities and source locator (`shard`, `row_index`, `qa_index`). Questions have textual `<image>` markers removed because the official processor inserts image tokens from image content.

Original image bytes are SHA-256 hashed. A single-image row uses the image SHA-256 directly. A multi-image row uses a versioned ordered digest over the individual image hashes and also retains every individual hash.

Manifests store only IDs, source locators, and non-clinical categorization metadata. They do not store image bytes, questions, or answers. The dataset reloads those from the source Parquet and refuses a manifest whose example/image identity no longer matches.

## Leakage control

Splitting flattened QA rows is prohibited. The builder unions all records that share any image hash. This handles the subtle case where two different multi-image records overlap by one image. Connected components are assigned deterministically to train/validation, with a source/task stratum used where possible.

The scale subsets are produced from a single seeded ordering of connected image groups, so they are nested:

```text
1% ⊂ 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%
```

Optional `--ood-source SOURCE` arguments remove those sources before the internal split and create `ood_val.jsonl`. Any image overlap between OOD and internal records causes a hard failure.

## Integrity and exclusions

The builder checks missing/unreadable images, empty text, extreme dimensions/text, exact duplicate QA pairs, and basic grounding validity. Every exclusion is counted and written to `exclusions.jsonl`; nothing is silently removed. Duplicate image hashes are counted but are not exclusions because multiple instructions per image are expected.

Grounding coordinates use SonoInstruct's `[0,1000]` convention. The evaluator
accepts exactly four finite coordinates, normalizes them mathematically to
`[0,1]`, and reports malformed, range, and area failures explicitly. No model
output or training target is changed by this scoring normalization.
