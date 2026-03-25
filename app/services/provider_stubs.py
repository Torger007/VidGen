import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.models.schemas import AdapterSignal, ProviderArtifact
from app.services.condition_preprocessors import (
    DepthAnythingPreprocessor,
    DepthEstimatorUnavailableError,
    OpenPosePreprocessor,
    OpenPoseUnavailableError,
)


#当前很多 provider 还是 stub，但接口已经分出了 openpose、depth、camera、transition 这些能力
class BaseProviderStub:
    provider_name = "base"

    def execute(
        self,
        *,
        signal: AdapterSignal,
        step_index: int,
        signal_index: int,
        root_dir: Path,
        source_image_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ProviderArtifact:
        raise NotImplementedError

    def _artifact_path(self, root_dir: Path, step_index: int, signal_index: int) -> Path:
        provider_dir = root_dir / self.provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir / f"step-{step_index:02d}-signal-{signal_index:02d}.json"

    def _json_write(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _asset_path(self, root_dir: Path, step_index: int, signal_index: int, suffix: str) -> Path:
        provider_dir = root_dir / self.provider_name
        provider_dir.mkdir(parents=True, exist_ok=True)
        return provider_dir / f"step-{step_index:02d}-signal-{signal_index:02d}{suffix}"

    def _resolve_source_image(
        self,
        source_image_path: str | None,
        frame_size: tuple[int, int] | None,
    ) -> Image.Image | None:
        if source_image_path is None:
            return None
        path = Path(source_image_path)
        if not path.exists():
            return None
        image = Image.open(path).convert("RGB")
        if frame_size:
            image = image.resize(frame_size)
        return image


class OpenPoseProviderStub(BaseProviderStub):
    provider_name = "openpose"

    def __init__(self) -> None:
        self._preprocessor = OpenPosePreprocessor()

    def execute(
        self,
        *,
        signal: AdapterSignal,
        step_index: int,
        signal_index: int,
        root_dir: Path,
        source_image_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ProviderArtifact:
        path = self._artifact_path(root_dir, step_index, signal_index)
        frame_window = signal.provider_payload.get("frame_window", [0, 0])
        pose_assets: list[dict[str, str | int]] = []
        skeleton_sequence = [
            {
                "frame_index": frame_index,
                "skeleton_id": 1,
                "joints": {
                    "head": [0.5, 0.18],
                    "left_hand": [0.35, 0.45],
                    "right_hand": [0.65, 0.45],
                    "left_foot": [0.42, 0.9],
                    "right_foot": [0.58, 0.9],
                },
                "confidence": round(max(0.4, signal.strength), 2),
            }
            for frame_index in range(int(frame_window[0]), int(frame_window[1]) + 1)
        ]
        mode = "stub"
        source_image = self._resolve_source_image(source_image_path, frame_size)
        if source_image is not None:
            try:
                pose_map = self._preprocessor.process(source_image)
                pose_asset_path = self._asset_path(root_dir, step_index, signal_index, ".png")
                pose_map.save(pose_asset_path)
                pose_assets = [
                    {
                        "frame_index": frame_index,
                        "image_path": str(pose_asset_path),
                    }
                    for frame_index in range(int(frame_window[0]), int(frame_window[1]) + 1)
                ]
                skeleton_sequence = [
                    {
                        "frame_index": frame_index,
                        "skeleton_id": 1,
                        "joints": {},
                        "confidence": round(max(0.4, signal.strength), 2),
                    }
                    for frame_index in range(int(frame_window[0]), int(frame_window[1]) + 1)
                ]
                mode = "openpose-detector"
            except (OpenPoseUnavailableError, Exception):
                pass
        payload = {
            "provider": self.provider_name,
            "artifact_type": "skeleton_sequence",
            "source": signal.source,
            "strength": signal.strength,
            "payload": signal.provider_payload,
            "mode": mode,
            "pose_assets": pose_assets,
            "skeleton_sequence": skeleton_sequence,
        }
        self._json_write(path, payload)
        return ProviderArtifact(
            provider=self.provider_name,
            adapter_type=signal.adapter_type,
            artifact_type="skeleton_sequence",
            step_index=step_index,
            signal_index=signal_index,
            artifact_path=str(path),
            summary=f"Generated skeleton sequence for {signal.source}",
        )


class DepthProviderStub(BaseProviderStub):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self._preprocessor = DepthAnythingPreprocessor() if provider_name == "depth-anything" else None

    def execute(
        self,
        *,
        signal: AdapterSignal,
        step_index: int,
        signal_index: int,
        root_dir: Path,
        source_image_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ProviderArtifact:
        path = self._artifact_path(root_dir, step_index, signal_index)
        frame_window = signal.provider_payload.get("frame_window", [0, 0])
        mode = "stub"
        depth_asset_path: str | None = None
        depth_array_path: str | None = None
        source_image = self._resolve_source_image(source_image_path, frame_size)
        if source_image is not None and self._preprocessor is not None:
            try:
                depth_image, depth_array = self._preprocessor.process(source_image)
                image_path = self._asset_path(root_dir, step_index, signal_index, ".png")
                depth_image.save(image_path)
                depth_asset_path = str(image_path)
                if depth_array is not None:
                    array_path = self._asset_path(root_dir, step_index, signal_index, ".npy")
                    np.save(array_path, depth_array)
                    depth_array_path = str(array_path)
                mode = "depth-anything"
            except (DepthEstimatorUnavailableError, Exception):
                pass
        frames = [
            {
                "frame_index": frame_index,
                "depth_asset": depth_asset_path or f"{self.provider_name}/step-{step_index:02d}/frame-{frame_index:03d}.npy",
                "depth_array_asset": depth_array_path,
                "focus_mode": signal.provider_payload.get("focus_mode", "scene"),
            }
            for frame_index in range(int(frame_window[0]), int(frame_window[1]) + 1)
        ]
        payload = {
            "provider": self.provider_name,
            "artifact_type": "depth_manifest",
            "source": signal.source,
            "strength": signal.strength,
            "payload": signal.provider_payload,
            "mode": mode,
            "frames": frames,
        }
        self._json_write(path, payload)
        return ProviderArtifact(
            provider=self.provider_name,
            adapter_type=signal.adapter_type,
            artifact_type="depth_manifest",
            step_index=step_index,
            signal_index=signal_index,
            artifact_path=str(path),
            summary=f"Generated depth manifest for {signal.source}",
        )


class CameraProviderStub(BaseProviderStub):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def execute(
        self,
        *,
        signal: AdapterSignal,
        step_index: int,
        signal_index: int,
        root_dir: Path,
        source_image_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ProviderArtifact:
        path = self._artifact_path(root_dir, step_index, signal_index)
        trajectory = {
            "path": signal.provider_payload.get("camera_path", "guided"),
            "start_frame": signal.provider_payload.get("start_frame", 0),
            "end_frame": signal.provider_payload.get("end_frame", 0),
            "intensity": signal.provider_payload.get("intensity", signal.strength),
            "mode": signal.provider_payload.get("camera_mode", "default"),
        }
        payload = {
            "provider": self.provider_name,
            "artifact_type": "camera_manifest",
            "source": signal.source,
            "strength": signal.strength,
            "payload": signal.provider_payload,
            "trajectory": trajectory,
        }
        self._json_write(path, payload)
        return ProviderArtifact(
            provider=self.provider_name,
            adapter_type=signal.adapter_type,
            artifact_type="camera_manifest",
            step_index=step_index,
            signal_index=signal_index,
            artifact_path=str(path),
            summary=f"Generated camera manifest for {signal.source}",
        )


class TransitionProviderStub(BaseProviderStub):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def execute(
        self,
        *,
        signal: AdapterSignal,
        step_index: int,
        signal_index: int,
        root_dir: Path,
        source_image_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ProviderArtifact:
        path = self._artifact_path(root_dir, step_index, signal_index)
        manifest = {
            "boundary_frame": signal.provider_payload.get("boundary_frame", 0),
            "blend_frames": signal.provider_payload.get("blend_frames", 1),
            "transition_type": signal.provider_payload.get("transition_type", signal.source),
            "provider": self.provider_name,
        }
        payload = {
            "provider": self.provider_name,
            "artifact_type": "transition_manifest",
            "source": signal.source,
            "strength": signal.strength,
            "payload": signal.provider_payload,
            "manifest": manifest,
        }
        self._json_write(path, payload)
        return ProviderArtifact(
            provider=self.provider_name,
            adapter_type=signal.adapter_type,
            artifact_type="transition_manifest",
            step_index=step_index,
            signal_index=signal_index,
            artifact_path=str(path),
            summary=f"Generated transition manifest for {signal.source}",
        )
