# Third-party models and data

SonoMed-VLM code is licensed under Apache-2.0. The model and dataset used by the
project remain governed by their own terms:

- **MedGemma 1.5 4B IT** (`google/medgemma-1.5-4b-it`) is provided under the
  [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms)
  and its incorporated [Prohibited Use Policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy).
  MedGemma is open-weight, not Apache-2.0-licensed model weights.
- **SonoInstruct** (`Ssdaizi/SonoInstruct`) is published under Apache-2.0. The
  dataset is not redistributed in this repository.

The separately published LoRA adapter is a MedGemma model derivative and is
distributed subject to the HAI-DEF terms. Users must obtain the gated MedGemma
base model independently and accept its terms before loading the adapter.
