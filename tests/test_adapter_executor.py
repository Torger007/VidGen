import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.adapter_executor import AdapterExecutor
from app.services.control_plan import ControlPlanBuilder
from app.services.control_signal_mapper import ControlSignalMapper
from app.services.prompting import PromptOrchestrator


def test_adapter_executor_writes_provider_artifacts(tmp_path: Path) -> None:
    bundle = PromptOrchestrator().build_bundle(
        "A robot walking through a city street at night",
        generation_profile="balanced",
    )
    control_plan = ControlPlanBuilder().build(bundle, "balanced", fps=8)
    adapter_plan = ControlSignalMapper().build(control_plan, "mock-svd", use_mock=True)
    execution = AdapterExecutor().execute(job_id="job-1", adapter_plan=adapter_plan, root_dir=tmp_path)

    assert execution.steps
    first_artifact = execution.steps[0].artifacts[0]
    assert Path(first_artifact.artifact_path).exists()
    assert first_artifact.provider
    assert first_artifact.artifact_type in {"skeleton_sequence", "depth_manifest", "camera_manifest", "transition_manifest"}
    assert execution.artifact_index


def test_adapter_executor_uses_real_pose_and_depth_preprocessors_when_available(tmp_path: Path) -> None:
    bundle = PromptOrchestrator().build_bundle(
        "A robot walking through a city street at night",
        generation_profile="balanced",
    )
    control_plan = ControlPlanBuilder().build(bundle, "balanced", fps=8)
    adapter_plan = ControlSignalMapper().build(control_plan, "mock-svd", use_mock=True)
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (64, 64), "white").save(reference_path)

    executor = AdapterExecutor()
    openpose = executor._registry["openpose"]
    depth = executor._registry["depth-anything"]
    openpose._preprocessor.process = lambda image: Image.new("RGB", image.size, "black")
    depth._preprocessor.process = lambda image: (Image.new("RGB", image.size, "gray"), np.ones((64, 64), dtype="float32"))

    execution = executor.execute(
        job_id="job-real",
        adapter_plan=adapter_plan,
        root_dir=tmp_path,
        source_image_path=str(reference_path),
        frame_size=(64, 64),
    )

    pose_artifact_path = Path(execution.artifact_index["skeleton_sequence"][0])
    pose_payload = json.loads(pose_artifact_path.read_text(encoding="utf-8"))
    assert pose_payload["mode"] == "openpose-detector"
    assert Path(pose_payload["pose_assets"][0]["image_path"]).exists()

    depth_artifact_path = Path(execution.artifact_index["depth_manifest"][0])
    depth_payload = json.loads(depth_artifact_path.read_text(encoding="utf-8"))
    assert depth_payload["mode"] == "depth-anything"
    assert Path(depth_payload["frames"][0]["depth_asset"]).exists()
    assert Path(depth_payload["frames"][0]["depth_array_asset"]).exists()


def test_adapter_executor_falls_back_to_stub_when_openpose_preprocessor_errors(tmp_path: Path) -> None:
    bundle = PromptOrchestrator().build_bundle(
        "A robot walking through a city street at night",
        generation_profile="balanced",
    )
    control_plan = ControlPlanBuilder().build(bundle, "balanced", fps=8)
    adapter_plan = ControlSignalMapper().build(control_plan, "mock-svd", use_mock=True)
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (64, 64), "white").save(reference_path)

    executor = AdapterExecutor()
    openpose = executor._registry["openpose"]
    openpose._preprocessor.process = lambda image: (_ for _ in ()).throw(
        AttributeError("module 'mediapipe' has no attribute 'solutions'")
    )

    execution = executor.execute(
        job_id="job-openpose-fallback",
        adapter_plan=adapter_plan,
        root_dir=tmp_path,
        source_image_path=str(reference_path),
        frame_size=(64, 64),
    )

    pose_artifact_path = Path(execution.artifact_index["skeleton_sequence"][0])
    pose_payload = json.loads(pose_artifact_path.read_text(encoding="utf-8"))
    assert pose_payload["mode"] == "stub"
    assert pose_payload["pose_assets"] == []
