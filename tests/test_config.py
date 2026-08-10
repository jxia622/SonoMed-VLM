from pathlib import Path

from sonomed_vlm.config import load_config


def test_config_inheritance_and_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SONOINSTRUCT_ROOT", str(tmp_path / "dataset"))
    monkeypatch.setenv("SONOMED_OUTPUT_ROOT", str(tmp_path / "outputs"))
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    assert config.model.name == "google/medgemma-1.5-4b-it"
    assert config.training.max_steps == 20
    assert config.training.learning_rate == 1e-4
    assert config.data.root == tmp_path / "dataset"
    assert config.output.root == tmp_path / "outputs"
    assert config.performance.profiler_enabled is False
    assert config.performance.timing_warmup_steps == 0
    assert config.performance.torch_compile is False
