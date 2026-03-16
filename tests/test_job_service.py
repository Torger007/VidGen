from pathlib import Path

from app.models.schemas import GenerateVideoRequest
from app.services.job_service import JobService


def test_create_job_in_eager_mode_writes_output(monkeypatch: object, tmp_path: Path) -> None:
    monkeypatch.setenv("VIDGEN_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDGEN_USE_MOCK_PIPELINE", "true")
    monkeypatch.setenv("VIDGEN_TASK_MODE", "eager")

    from app.core.config import get_settings

    get_settings.cache_clear()
    service = JobService()
    job = service.create_job(
        GenerateVideoRequest(prompt="A robot walking through a rainy neon city street at night")
    )

    saved = service.get_job(job.job_id)
    assert saved is not None
    assert saved.status == "succeeded"
    assert saved.output_path is not None
    assert Path(saved.output_path).exists()


def test_create_job_with_multiple_candidates_records_candidate_paths(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIDGEN_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDGEN_USE_MOCK_PIPELINE", "true")
    monkeypatch.setenv("VIDGEN_TASK_MODE", "eager")

    from app.core.config import get_settings

    get_settings.cache_clear()
    service = JobService()
    job = service.create_job(
        GenerateVideoRequest(
            prompt="A cinematic robot walking through a rainy neon city street at night",
            parameters={"num_candidates": 3},
        )
    )

    saved = service.get_job(job.job_id)
    assert saved is not None
    assert saved.status == "succeeded"
    assert len(saved.candidate_paths) == 3
    assert all(Path(path).exists() for path in saved.candidate_paths)
    assert saved.selected_candidate is not None
    assert saved.selected_candidate.evaluation is not None
    assert saved.candidates[0].evaluation is not None
    assert job.selected_candidate is not None
    assert saved.selection_mode is not None
    assert saved.generation_profile is not None
