# RSI and robustness experiments

## Iterative self-training from 5% Qwen SFT

We tested CoEvol-style alternating critic and generator training, referred to as RSI in the project. The aim was to use additional unlabeled image-question pairs to close the gap between a 5% supervised warm start and full SFT.

The rank-8 Qwen warm start used 5% labeled data. Each of four rounds used 979 disjoint labeled open-response critic seeds and 3,000 fresh questions from a cleaned 74,389-question pool, excluding image connections to the 5% seed and the selector-gate questions. References for the unlabeled pool were hidden. The post-critic model sampled three answers per question; the critic arm kept deduplicated answers with P(M) > 0.55 and continued SFT on the same adapter.

The random arm also included critic training, but randomly selected eligible answers in counts determined by its own critic's acceptance count. Thus it controls selection within the evolving pipeline; the two arms do not have identical accepted counts or trajectories. Provenance labels distinguish reference from generated answers, not verified correctness.

### Completed results

Open-response multilingual BERTScore F1 on 434 development examples:

| Round | Critic filtering | Random selection |
|---|---:|---:|
| 0: 5% warm start | 0.7694 | 0.7694 |
| 1 | 0.7580 | 0.7630 |
| 2 | 0.7474 | 0.7549 |
| 3 | 0.7358 | 0.7416 |
| 4 | 0.7202 | 0.7350 |

The separate full-SFT reference scored 0.7835. Neither arm improved the seed; both declined each round, and critic filtering was consistently below random selection. This recipe did not close the supervised-data gap. At round 4, the critic arm retained 92.0% MCQ accuracy and 65.2% Localization@0.5 on the small retention subsets.

One seed, no confidence intervals, and only 37 reports limit interpretation. Nineteen report references have verbatim training copies. BERTScore measures semantic similarity, not clinical correctness. The completed campaign is a negative result, superseding earlier plans that described the four rounds as future work.

## Image reliance and controlled noise

Completed November 16, 2026 on a frozen Qwen checkpoint. From 2,093 eligible validation image groups, 1,000 were sampled with source/organ/task stratification, one single-image anatomy-recognition or lesion-diagnosis question per group. All seven conditions retained the same question, choices, reference, and greedy decoding settings.

| Condition | Weighted accuracy / original-answer agreement | Drop from original (percentage points) | Paired 95% CI for drop |
|---|---:|---:|---:|
| Original image | 97.55% | 0.00 | [0.00, 0.00] |
| Gaussian noise, σ=0.08 | 96.95% | 0.60 | [-0.20, 1.41] |
| Gaussian noise, σ=0.18 | 95.65% | 1.90 | [0.92, 2.88] |
| Gaussian noise, σ=0.38 | 88.38% | 9.17 | [7.48, 10.89] |
| Fixed gray image | 43.11% | 54.44 | [51.99, 57.02] |
| Mismatched image: original-answer agreement | 84.97% | 12.58 | [11.08, 14.09] |
| Text only | 57.07% | 40.48 | [37.95, 43.20] |

Intervals use 5,000 stratified paired image-group bootstrap replicates. Original versus σ=0.38 was the prespecified primary comparison; other contrasts are descriptive. The lowest-noise interval includes zero. Noise was added after native processor resizing and before normalization; image conditions preserved visual-token geometry. Seed: 20261116; greedy generation, maximum 256 new tokens, batch size four.

All 7,000 responses were complete and parseable. Strict letter and semantic-choice accuracies agreed. The integrity audit reproduced scores and verified unchanged in-memory model and on-disk checkpoint hashes across evaluation.

Mismatched images often depict the same organ, so 84.97% is agreement with the **original** answer, not correctness for the replacement image. An exploratory 63-question subset with unchanged questions/options but different donor-image labels had 89.06% original-image accuracy, 9.37% original-answer agreement after replacement, and 90.63% agreement with the donor label. Donor reuse (47 distinct donor images) limits independent-sample interpretation.

The model uses visual evidence under these controls, while the 57.07% text-only score shows a substantial contribution from textual clues or priors. These tests do not distinguish tissue evidence from overlays or measurement markers, establish patient independence, or demonstrate robustness across institutions, devices, or acquisition protocols.

See [source provenance and protocol boundaries](STUDY.md). Raw images, questions, per-example generations, and cluster artifacts are not included in this public summary.
