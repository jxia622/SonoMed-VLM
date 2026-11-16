# Interview-deck figures

Extracted from the inline SVG charts in slides 3 and 4 of `Jack_Xia_Interview_Deck.html`, supplied by Jack Xia. Bar positions, points, values, axes, and colors are retained. Standalone titles, legends, white backgrounds, and protocol notes replace the surrounding slide layout; unrelated slides and branding are omitted.

- [Three-model comparison](three-model-comparison.svg): before/after 5% SFT. MedGemma and Qwen historical semantic-choice baselines differ in scoring and item count from the corrected after-SFT answer-text results. The shared before-count annotation applies to those historical baselines; do not infer a controlled improvement from those paired bars. Baseline boxes use the presentation's corrected values, not the earlier September inventory.
- [Historical scaling](historical-data-scaling.svg): the historical v1 choices-omitted series with 4,905 MCQ-source items, 4,628 ordinary QA/open items, and 565 boxes. This is distinct from the corrected open_qa_v2 comparison. The chart's semantic-choice scores are answer-match diagnostics. One seed per point; no confidence intervals.

The [study guide](../../STUDY.md) explains protocol boundaries. SVG files retain vector text and scale without raster blur on GitHub.
