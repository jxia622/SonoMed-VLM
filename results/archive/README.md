# Historical results — superseded for current comparisons

These files preserve the v1 training/evaluation recipe, whose MCQ choices were omitted.
Their answer-content matching scores are not standard MCQ accuracy and are not comparable
to v2 open-QA exact match. Do not mix validation denominators or connect these points to
v2 learning curves. Current corrected results: [JSON](../research_results.json),
[CSV](../research_results.csv). The published Hugging Face adapter still contains v1 weights.
The legacy exporter is `scripts/build_research_results.py`; direct its output to a separate
historical directory. The v2 exporter is `scripts/summarize_open_qa.py`.
