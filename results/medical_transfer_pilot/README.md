# Completed medical intermediate-training pilot

Source: the verified CRC summary for `medical-transfer-20260929`, completed 30 September 2026.
Seed 42 only. See [the report](../../docs/MEDICAL_INTERMEDIATE_TRANSFER.md) for methods,
all nine ultrasound comparisons, text-knowledge diagnostics and interpretation.

- `results.json`: unchanged CRC aggregate; all 27 downstream domain evaluations,
  six original/intermediate diagnostic evaluations, subgroup scores, downstream
  prediction hashes, paired arm differences and null single-seed standard deviations.
- `results.md`: unchanged CRC headline summary.
- `seed_summary.csv`: CRC long-format downstream metrics; line endings normalized to LF.
- `evaluation_metrics.csv`: generated flat table of all 33 evaluations, including
  medical/general diagnostics, for plotting and independent inspection.
- `provenance.json`: source-file checksums, training-source commit and successful jobs.

Regenerate the flat CSV and all three PNG/SVG/PDF figures with:
`python scripts/plot_medical_transfer_results.py`.
These are internal ultrasound validation and open-ended text diagnostics, not MCQ
accuracy, official SonoBench scores, or a test of human-like medical reasoning.
