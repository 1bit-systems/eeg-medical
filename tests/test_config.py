"""Config loading must tolerate the shipped YAML (which carries ZUNA config extras)."""

from pathlib import Path

from eeg_medical.training.config import FinetuneConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_from_yaml_ships_config():
    cfg = FinetuneConfig.from_yaml(REPO_ROOT / "configs" / "tuh_clinical_finetune.yaml")
    assert cfg.total_steps == 50000
    assert cfg.learning_rate == 1.0e-4
    assert "denoising" in cfg.clinical_tasks
    assert cfg.data_dir == "data/tuh_processed"
