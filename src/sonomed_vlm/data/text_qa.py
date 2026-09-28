"""Explicit text-only QA manifests for controlled intermediate training."""

from sonomed_vlm.data.schema import NormalizedExample
from sonomed_vlm.utils.io import read_jsonl, sha256_json

SYSTEM = "Answer the question with the answer text only. Be concise. Do not output an option letter or a list of choices."


class TextQADataset:
    def __init__(self, manifest, data_root=None, **kwargs):
        self.records = list(read_jsonl(manifest))
        ids = [r["example_id"] for r in self.records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicated text QA manifest")
        for r in self.records:
            if not r["question"].strip() or not r["answer"].strip() or r.get("options"):
                raise ValueError("Text QA must contain standalone questions and answer text")
            if r["content_sha256"] != sha256_json({"q": r["question"], "a": r["answer"]}):
                raise ValueError("Text QA content digest mismatch")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return NormalizedExample(
            example_id=r["example_id"],
            image_id="text:" + r["example_id"],
            images=[],
            source_dataset=r["source"],
            task_family="text_knowledge",
            task_type="open_qa",
            focus=None,
            system_prompt=SYSTEM,
            user_prompt=r["question"],
            assistant_response=r["answer"],
            metadata={"options": [], "answer_label": None},
        )


def dataset_class(config):
    from sonomed_vlm.data.sonoinstruct import ManifestDataset

    return TextQADataset if config.data.format == "text_qa" else ManifestDataset
