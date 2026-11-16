# Study guide and evidence provenance

Updated November 16, 2026. The opening ultrasound slides of Jack Xia's interview deck define the study narrative: ultrasound-domain VLM development, three-model adaptation, scaling, then RSI. Later completed experiment records supply the RSI outcomes and robustness results.

## Questions

- Which starting model transfers best to ultrasound tasks with limited supervision?
- How does task performance change as labeled data increases?
- Can iterative self-training close the gap between 5% and full supervised training?
- Does the resulting model rely on images, and how do controlled perturbations affect it?

MedGemma motivated the initial medical-model baseline. Gemma tests the intuition that general-purpose initialization might avoid modality-specific specialization. Qwen supplies a strong general-purpose alternative. Cross-family results alone cannot identify the causal effect of medical pretraining. The separate [medical intermediate-training pilot](MEDICAL_INTERMEDIATE_TRANSFER.md) offers a controlled Qwen experiment and found no consistent overall benefit in one seed.

## Keep experiment families separate

| Family | Evaluation population | Interpretation |
|---|---|---|
| Historical v1 scaling | 10,098: 4,905 MCQ-source, 4,628 QA/open, 565 boxes | Choices-omitted, option-aware answer-match diagnostics; archived, not ordinary MCQ accuracy |
| Corrected open_qa_v2 scaling and 5% Gemma comparison | 9,964: 4,938 converted answer-text QA, 4,461 ordinary QA/open, 565 boxes | Primary README model comparison; no options supplied or used to resolve predicted letters |
| RSI development | 1,024: 524 MCQ, 434 open, 66 boxes | Separate rank-8 warm start and iterative-training comparison |
| Image reliance | 1,000 sampled eligible image groups, seven conditions | Choices-present anatomy/lesion questions; weighted, paired analysis |

The opening slides show historical semantic-choice baselines beside corrected answer-text results for MedGemma and Qwen. Those are not controlled before/after gains. Their corrected baseline grounding numbers also differ from the older inventory. The README therefore reports the matched corrected **after-SFT** comparison and keeps historical scaling in its existing archive rather than constructing a mixed baseline table.

## Source records

The published September scaling evidence remains in [research_results.json](../results/research_results.json), [research report](RESEARCH.md), and [protocol correction](OPEN_QA_V2.md).

The new summaries transcribe aggregate findings from these local research records (not publicly bundled raw artifacts):

- Gemma: `artifacts/gemma3_5pct_r16_20261110/results/comparison.json`; inventory `docs/MODEL_RESULTS_INVENTORY_20261110.md` and interview slide 3.
- RSI: `docs/COEVOL_5PCT_CLEAN_20261109.md`, completed-results section dated November 10.
- Robustness: `reports/image_reliance_1000_20261116.md`, completed November 16; aggregate `artifacts/image_reliance_1000_20261116/records/campaign/results.json`.

These local paths document provenance, not downloadable public files. This release updates the public research narrative and summaries; it does not claim to publish every local experiment implementation or independently rerun GPU experiments.
