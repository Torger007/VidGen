import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from app.models.schemas import GenerationContext


#负责把 pose/depth/camera 等条件真正注入 image/video pipeline 调用参数。
@dataclass
class ConditionRouting:
    image_kwargs: dict[str, Any] = field(default_factory=dict)
    video_kwargs: dict[str, Any] = field(default_factory=dict)
    pose_control_image: Image.Image | None = None
    depth_control_image: Image.Image | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    camera_paths: list[dict[str, Any]] = field(default_factory=list)
    used_image_conditioning: bool = False
    used_video_conditioning: bool = False
    image_call_summary: dict[str, Any] = field(default_factory=dict)
    video_call_summary: dict[str, Any] = field(default_factory=dict)


class ConditionRouter:
    def build(
        self,
        generation_context: GenerationContext | None,
        *,
        width: int,
        height: int,
    ) -> ConditionRouting:
        if generation_context is None:
            return ConditionRouting()

        metadata = generation_context.metadata
        routing = ConditionRouting()
        pose_paths = self._metadata_paths(metadata.get("pose_artifact_paths"))
        depth_paths = self._metadata_paths(metadata.get("depth_artifact_paths"))
        transition_paths = self._metadata_paths(metadata.get("transition_artifact_paths"))
        camera_paths = self._metadata_paths(metadata.get("camera_artifact_paths"))

        if pose_paths:
            routing.pose_control_image = self._build_pose_control_image(Path(pose_paths[0]), width, height)
        if depth_paths:
            routing.depth_control_image = self._build_depth_control_image(Path(depth_paths[0]), width, height)
        if transition_paths:
            routing.transitions = [self._load_transition(Path(path)) for path in transition_paths]
        if camera_paths:
            routing.camera_paths = [self._load_camera(Path(path)) for path in camera_paths]
        return routing

    def apply_image_branch(self, pipe: Any, call_kwargs: dict[str, Any], routing: ConditionRouting) -> None:
        params = self._callable_params(pipe)
        control_images, scales = self._control_images_and_scales(routing)
        if control_images:
            applied = self._inject_control_images(call_kwargs, params, control_images, scales)
            routing.used_image_conditioning = applied
        routing.image_call_summary = self._summarize_call_kwargs(call_kwargs, routing, branch="image")

    def apply_video_branch(self, pipe: Any, call_kwargs: dict[str, Any], routing: ConditionRouting) -> None:
        params = self._callable_params(pipe)
        control_images, scales = self._control_images_and_scales(routing)
        applied = False
        if control_images:
            applied = self._inject_control_images(call_kwargs, params, control_images, scales)
        camera_intensity = self._camera_intensity(routing.camera_paths)
        if camera_intensity > 0:
            if "motion_bucket_id" in params:
                call_kwargs["motion_bucket_id"] = min(255, 96 + round(camera_intensity * 96))
                applied = True
            if "noise_aug_strength" in params:
                call_kwargs["noise_aug_strength"] = min(0.2, round(0.02 + camera_intensity * 0.08, 3))
                applied = True
        routing.used_video_conditioning = applied
        routing.video_call_summary = self._summarize_call_kwargs(call_kwargs, routing, branch="video")

    def summarize(self, routing: ConditionRouting) -> dict[str, Any]:
        return {
            "pose_control": routing.pose_control_image is not None,
            "depth_control": routing.depth_control_image is not None,
            "transition_count": len(routing.transitions),
            "camera_path_count": len(routing.camera_paths),
            "used_image_conditioning": routing.used_image_conditioning,
            "used_video_conditioning": routing.used_video_conditioning,
            "image_call_summary": routing.image_call_summary,
            "video_call_summary": routing.video_call_summary,
        }

    def apply_initial_frame_fallback(self, image: Image.Image, routing: ConditionRouting) -> Image.Image:
        output = image.convert("RGB")
        if routing.pose_control_image and not routing.used_image_conditioning:
            pose_overlay = routing.pose_control_image.resize(output.size).convert("RGB")
            output = Image.blend(output, pose_overlay, 0.12)
        if routing.depth_control_image and not routing.used_image_conditioning:
            depth_overlay = routing.depth_control_image.resize(output.size).convert("RGB")
            output = ImageChops.soft_light(output, depth_overlay)
        return output

    def apply_video_postprocess(self, frames: list[Image.Image], routing: ConditionRouting) -> list[Image.Image]:
        processed = [frame.convert("RGB") for frame in frames]
        if routing.camera_paths and not routing.used_video_conditioning:
            processed = self._apply_camera_motion(processed, routing.camera_paths)
        if routing.transitions:
            processed = self._apply_transition_blends(processed, routing.transitions)
        return processed

    def _inject_control_images(
        self,
        call_kwargs: dict[str, Any],
        params: set[str],
        control_images: list[Image.Image],
        scales: list[float],
    ) -> bool:
        if not control_images:
            return False
        applied = False
        if "control_images" in params:
            call_kwargs["control_images"] = control_images
            applied = True
        elif "control_image" in params:
            call_kwargs["control_image"] = control_images if len(control_images) > 1 else control_images[0]
            applied = True
        elif "conditioning_images" in params:
            call_kwargs["conditioning_images"] = control_images
            applied = True
        elif "conditioning_image" in params:
            call_kwargs["conditioning_image"] = control_images if len(control_images) > 1 else control_images[0]
            applied = True

        if applied:
            if "controlnet_conditioning_scale" in params:
                call_kwargs["controlnet_conditioning_scale"] = scales if len(scales) > 1 else scales[0]
            elif "conditioning_scale" in params:
                call_kwargs["conditioning_scale"] = scales if len(scales) > 1 else scales[0]
        return applied

    def _summarize_call_kwargs(
        self,
        call_kwargs: dict[str, Any],
        routing: ConditionRouting,
        *,
        branch: str,
    ) -> dict[str, Any]:
        control_value = (
            call_kwargs.get("control_images")
            or call_kwargs.get("control_image")
            or call_kwargs.get("conditioning_images")
            or call_kwargs.get("conditioning_image")
        )
        if isinstance(control_value, list):
            control_count = len(control_value)
            control_sizes = [getattr(item, "size", None) for item in control_value]
        elif control_value is not None:
            control_count = 1
            control_sizes = [getattr(control_value, "size", None)]
        else:
            control_count = 0
            control_sizes = []
        return {
            "branch": branch,
            "has_control_input": control_count > 0,
            "control_image_count": control_count,
            "control_image_sizes": control_sizes,
            "controlnet_conditioning_scale": call_kwargs.get("controlnet_conditioning_scale")
            or call_kwargs.get("conditioning_scale"),
            "motion_bucket_id": call_kwargs.get("motion_bucket_id"),
            "noise_aug_strength": call_kwargs.get("noise_aug_strength"),
            "pose_control": routing.pose_control_image is not None,
            "depth_control": routing.depth_control_image is not None,
        }

    def _callable_params(self, pipe: Any) -> set[str]:
        try:
            signature = inspect.signature(pipe.__call__)
        except (TypeError, ValueError):
            return set()
        return set(signature.parameters)

    def _control_images_and_scales(self, routing: ConditionRouting) -> tuple[list[Image.Image], list[float]]:
        control_images: list[Image.Image] = []
        scales: list[float] = []
        if routing.pose_control_image:
            control_images.append(routing.pose_control_image)
            scales.append(0.75)
        if routing.depth_control_image:
            control_images.append(routing.depth_control_image)
            scales.append(0.7)
        return control_images, scales

    def _build_pose_control_image(self, path: Path, width: int, height: int) -> Image.Image:
        payload = self._load_json(path)
        pose_assets = payload.get("pose_assets", [])
        if pose_assets:
            image_path = pose_assets[0].get("image_path") if isinstance(pose_assets[0], dict) else None
            if image_path and Path(image_path).exists():
                return Image.open(image_path).convert("RGB").resize((width, height))
        canvas = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(canvas)
        skeletons = payload.get("skeleton_sequence", [])
        joint_pairs = [
            ("head", "left_hand"),
            ("head", "right_hand"),
            ("left_hand", "left_foot"),
            ("right_hand", "right_foot"),
            ("left_foot", "right_foot"),
        ]
        for skeleton in skeletons[:3]:
            joints = skeleton.get("joints", {})
            for start_name, end_name in joint_pairs:
                start = self._point(joints.get(start_name), width, height)
                end = self._point(joints.get(end_name), width, height)
                if start and end:
                    draw.line([start, end], fill=(255, 255, 255), width=4)
            for point in joints.values():
                joint = self._point(point, width, height)
                if joint:
                    radius = 5
                    draw.ellipse(
                        (joint[0] - radius, joint[1] - radius, joint[0] + radius, joint[1] + radius),
                        fill=(0, 180, 255),
                    )
        return canvas

    def _build_depth_control_image(self, path: Path, width: int, height: int) -> Image.Image:
        payload = self._load_json(path)
        frames = payload.get("frames", [])
        if frames:
            first_asset = frames[0].get("depth_asset") if isinstance(frames[0], dict) else None
            if first_asset and Path(first_asset).exists():
                return Image.open(first_asset).convert("RGB").resize((width, height))
        focus_mode = "scene"
        if frames:
            focus_mode = str(frames[0].get("focus_mode", "scene"))

        image = Image.new("L", (width, height))
        center_x = width / 2
        center_y = height / 2
        max_distance = max(center_x, center_y, 1)
        for y in range(height):
            for x in range(width):
                if focus_mode == "subject":
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    value = max(0, min(255, int(255 - (distance / max_distance) * 180)))
                else:
                    value = max(0, min(255, int(255 * (1 - y / max(height - 1, 1)))))
                image.putpixel((x, y), value)
        return image.convert("RGB")

    def _load_transition(self, path: Path) -> dict[str, Any]:
        payload = self._load_json(path)
        return payload.get("manifest", {})

    def _load_camera(self, path: Path) -> dict[str, Any]:
        payload = self._load_json(path)
        return payload.get("trajectory", {})

    def _apply_transition_blends(self, frames: list[Image.Image], transitions: list[dict[str, Any]]) -> list[Image.Image]:
        output = list(frames)
        total_frames = len(output)
        for transition in transitions:
            boundary = int(transition.get("boundary_frame", 0))
            blend_frames = max(1, int(transition.get("blend_frames", 1)))
            for offset in range(1, blend_frames + 1):
                left_index = boundary - offset
                right_index = boundary + offset - 1
                if left_index < 0 or right_index >= total_frames:
                    continue
                alpha = offset / (blend_frames + 1)
                output[right_index] = Image.blend(output[left_index], output[right_index], alpha)
        return output

    def _apply_camera_motion(self, frames: list[Image.Image], trajectories: list[dict[str, Any]]) -> list[Image.Image]:
        if not frames:
            return frames
        dominant = max(trajectories, key=lambda item: float(item.get("intensity", 0.0)))
        path = str(dominant.get("path", "static"))
        intensity = max(0.0, min(1.0, float(dominant.get("intensity", 0.0))))
        if path in {"static", "locked"} or intensity == 0:
            return frames

        output: list[Image.Image] = []
        width, height = frames[0].size
        crop_margin = int(min(width, height) * 0.08 * intensity)
        total = max(1, len(frames) - 1)
        for index, frame in enumerate(frames):
            progress = index / total
            left = 0
            top = 0
            right = width
            bottom = height
            if path in {"push-in", "dolly-in", "zoom-in"}:
                left += int(crop_margin * progress)
                top += int(crop_margin * progress)
                right -= int(crop_margin * progress)
                bottom -= int(crop_margin * progress)
            elif path in {"pull-out", "zoom-out"}:
                shrink = crop_margin - int(crop_margin * progress)
                left += shrink
                top += shrink
                right -= shrink
                bottom -= shrink
            elif path in {"pan-left", "track-left"}:
                shift = int(crop_margin * progress)
                left += shift
                right += shift
            elif path in {"pan-right", "track-right"}:
                shift = int(crop_margin * progress)
                left -= shift
                right -= shift

            left = max(0, min(left, width - 2))
            top = max(0, min(top, height - 2))
            right = max(left + 1, min(right, width))
            bottom = max(top + 1, min(bottom, height))
            output.append(frame.crop((left, top, right, bottom)).resize((width, height)))
        return output

    def _camera_intensity(self, trajectories: list[dict[str, Any]]) -> float:
        if not trajectories:
            return 0.0
        return max(max(0.0, min(1.0, float(item.get("intensity", 0.0)))) for item in trajectories)

    def _metadata_paths(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _point(self, raw: Any, width: int, height: int) -> tuple[int, int] | None:
        if not isinstance(raw, list) or len(raw) != 2:
            return None
        return int(float(raw[0]) * width), int(float(raw[1]) * height)
