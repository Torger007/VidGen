from app.core.generation_profiles import get_generation_profile
from app.models.schemas import GenerateVideoRequest
from app.services.job_service import JobService


def test_get_generation_profile_returns_defaults() -> None:
    name, defaults = get_generation_profile("strict")
    assert name == "strict"
    assert defaults["min_total_score"] >= 0.6


def test_job_service_applies_profile_defaults(monkeypatch: object, tmp_path: str) -> None:
    monkeypatch.setenv("VIDGEN_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDGEN_USE_MOCK_PIPELINE", "true")
    monkeypatch.setenv("VIDGEN_TASK_MODE", "eager")

    from app.core.config import get_settings

    get_settings.cache_clear()
    service = JobService()
    job = service.create_job(
        GenerateVideoRequest(
            prompt="A cinematic robot scene moving through neon rain",
            generation_profile="strict",
        )
    )

    assert job.generation_profile == "strict"
    assert job.parameters.min_total_score >= 0.6
    assert "stable" in job.prompt_bundle.style.lower()
    assert job.control_plan is not None
    assert job.parameters.num_frames == job.control_plan.total_frames
    assert len(job.control_plan.pose_hint_types) == len(job.control_plan.steps)


def test_request_parameters_override_profile(monkeypatch: object, tmp_path: str) -> None:
    monkeypatch.setenv("VIDGEN_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("VIDGEN_USE_MOCK_PIPELINE", "true")
    monkeypatch.setenv("VIDGEN_TASK_MODE", "eager")

    from app.core.config import get_settings

    get_settings.cache_clear()
    service = JobService()
    job = service.create_job(
        GenerateVideoRequest(
            prompt="A cinematic robot scene moving through neon rain",
            generation_profile="strict",
            parameters={"min_total_score": 0.2, "num_candidates": 1},
        )
    )

    assert job.parameters.min_total_score == 0.2
