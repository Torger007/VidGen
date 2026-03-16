from pathlib import Path

from app.services.adapter_executor import AdapterExecutor
from app.services.control_plan import ControlPlanBuilder
from app.services.control_signal_mapper import ControlSignalMapper
from app.services.middleware_consumer import MiddlewareConsumer
from app.services.prompting import PromptOrchestrator


def test_middleware_consumer_builds_generation_context(tmp_path: Path) -> None:
    bundle = PromptOrchestrator().build_bundle(
        "A robot walking through a city street at night",
        generation_profile="balanced",
    )
    control_plan = ControlPlanBuilder().build(bundle, "balanced", fps=8)
    adapter_plan = ControlSignalMapper().build(control_plan, "mock-svd", use_mock=True)
    execution = AdapterExecutor().execute(job_id="job-ctx", adapter_plan=adapter_plan, root_dir=tmp_path)
    context = MiddlewareConsumer().build_context(execution)

    assert context.prompt_suffixes
    assert context.decode_chunk_size >= 4
    assert context.metadata
    assert context.metadata["pose_artifact_paths"]
    assert context.metadata["depth_artifact_paths"]
    assert context.metadata["transition_artifact_paths"]
    assert context.metadata["camera_artifact_paths"]
    assert "depth_asset_images" in context.metadata
