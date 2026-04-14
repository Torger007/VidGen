from pathlib import Path


def test_settings_default_device_is_cpu(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setenv("VIDGEN_STORAGE_ROOT", str(tmp_path))
    monkeypatch.delenv("VIDGEN_DEVICE", raising=False)

    from app.core.config import Settings

    settings = Settings(_env_file=None)
    assert settings.device == "cpu"
