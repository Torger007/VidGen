from pathlib import Path

from app.models.schemas import AdapterPlan, ProviderExecutionPlan, ProviderExecutionStep
from app.services.provider_stubs import (
    CameraProviderStub,
    DepthProviderStub,
    OpenPoseProviderStub,
    TransitionProviderStub,
)


#执行这些adapter，并生成provider artifacts
class AdapterExecutor:
    def __init__(self) -> None:
        self._registry = {
            "openpose": OpenPoseProviderStub(),
            "depth-anything": DepthProviderStub("depth-anything"),
            "midas": DepthProviderStub("midas"),
            "depth-soft": DepthProviderStub("depth-soft"),
            "camera-motion-engine": CameraProviderStub("camera-motion-engine"),
            "static-camera-engine": CameraProviderStub("static-camera-engine"),
            "segment-stitcher": TransitionProviderStub("segment-stitcher"),
            "cut-stitcher": TransitionProviderStub("cut-stitcher"),
        }

    def execute(
        self,
        *,
        job_id: str,
        adapter_plan: AdapterPlan,
        root_dir: Path,
        source_image_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ProviderExecutionPlan:
        execution_root = root_dir / job_id
        execution_root.mkdir(parents=True, exist_ok=True)
        steps: list[ProviderExecutionStep] = []
        artifact_index: dict[str, list[str]] = {}
        for step in adapter_plan.steps:
            artifacts = []
            for signal_index, signal in enumerate(step.signals, start=1):
                stub = self._registry.get(signal.provider or "")
                if stub is None:
                    continue
                artifacts.append(
                    stub.execute(
                        signal=signal,
                        step_index=step.step_index,
                        signal_index=signal_index,
                        root_dir=execution_root,
                        source_image_path=source_image_path,
                        frame_size=frame_size,
                    )
                )
                artifact = artifacts[-1]
                artifact_index.setdefault(artifact.artifact_type, []).append(artifact.artifact_path)
            steps.append(ProviderExecutionStep(step_index=step.step_index, beat=step.beat, artifacts=artifacts))
        return ProviderExecutionPlan(
            job_id=job_id,
            execution_mode=adapter_plan.execution_mode,
            root_dir=str(execution_root),
            steps=steps,
            artifact_index=artifact_index,
        )
