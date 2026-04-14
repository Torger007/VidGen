import json
from pathlib import Path

from app.models.schemas import JobRecord
from app.services.job_store import JobStore


def _build_job_payload() -> dict:
    return {
        "job_id": "job-1",
        "prompt": "test prompt",
        "status": "queued",
        "prompt_bundle": {
            "subject": "robot",
            "scene": "city",
            "scene_template": "city",
            "style": "cinematic",
            "action": "walk",
            "camera": "slow dolly in",
            "negative_prompt": "blurry",
        },
        "parameters": {"model": "mock-svd"},
        "created_at": "2026-03-12T00:00:00Z",
        "updated_at": "2026-03-12T00:00:01Z",
    }


def test_job_store_load_retries_transient_partial_json(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setenv("VIDGEN_STORAGE_ROOT", str(tmp_path))

    from app.core.config import get_settings

    get_settings.cache_clear()
    store = JobStore()
    path = tmp_path / "jobs" / "job-1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_job_payload()), encoding="utf-8")

    original_read_text = Path.read_text
    calls = {"count": 0}

    def flaky_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == path and calls["count"] == 0:
            calls["count"] += 1
            return '{"job_id": "job-1"'
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    job = store.load("job-1")

    assert isinstance(job, JobRecord)
    assert job.job_id == "job-1"
    assert calls["count"] == 1
