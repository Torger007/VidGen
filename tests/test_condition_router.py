import json
from pathlib import Path

from PIL import Image

from app.models.schemas import GenerationContext
from app.services.condition_router import ConditionRouter


class FakeImageResult:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class FakeVideoResult:
    def __init__(self, frames: list[Image.Image]) -> None:
        self.frames = [frames]


class CapturingImagePipe:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
        control_image=None,
        controlnet_conditioning_scale=None,
        generator=None,
    ) -> FakeImageResult:
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "generator": generator,
            }
        )
        return FakeImageResult(Image.new("RGB", (width, height), "gray"))


class CapturingVideoPipe:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self,
        *,
        image: Image.Image,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        num_inference_steps: int,
        decode_chunk_size: int,
        control_image=None,
        controlnet_conditioning_scale=None,
        motion_bucket_id=None,
        noise_aug_strength=None,
    ) -> FakeVideoResult:
        self.calls.append(
            {
                "image": image,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "num_inference_steps": num_inference_steps,
                "decode_chunk_size": decode_chunk_size,
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "motion_bucket_id": motion_bucket_id,
                "noise_aug_strength": noise_aug_strength,
            }
        )
        frames = [
            Image.new("RGB", (width, height), color)
            for color in ("red", "blue", "green", "yellow")
        ]
        return FakeVideoResult(frames[:num_frames])


def test_video_pipeline_routes_conditions_into_inference_and_postprocess(tmp_path: Path) -> None:
    pose_path = _write_json(
        tmp_path / "pose.json",
        {
            "skeleton_sequence": [
                {
                    "frame_index": 0,
                    "joints": {
                        "head": [0.5, 0.2],
                        "left_hand": [0.35, 0.45],
                        "right_hand": [0.65, 0.45],
                        "left_foot": [0.42, 0.9],
                        "right_foot": [0.58, 0.9],
                    },
                }
            ]
        },
    )
    depth_path = _write_json(
        tmp_path / "depth.json",
        {"frames": [{"frame_index": 0, "focus_mode": "subject"}]},
    )
    transition_path = _write_json(
        tmp_path / "transition.json",
        {"manifest": {"boundary_frame": 1, "blend_frames": 1, "transition_type": "dissolve-short"}},
    )
    camera_path = _write_json(
        tmp_path / "camera.json",
        {"trajectory": {"path": "dolly-in", "intensity": 0.6, "start_frame": 0, "end_frame": 3}},
    )

    context = GenerationContext(
        pose_guidance="pose-conditioned motion continuity",
        depth_guidance="depth-aware composition (subject)",
        camera_guidance="camera path follows dolly-in",
        transition_guidance="smooth segment transition with 1 blend frames",
        metadata={
            "pose_artifact_paths": [str(pose_path)],
            "depth_artifact_paths": [str(depth_path)],
            "transition_artifact_paths": [str(transition_path)],
            "camera_artifact_paths": [str(camera_path)],
        },
    )
    router = ConditionRouter()
    image_pipe = CapturingImagePipe()
    video_pipe = CapturingVideoPipe()
    routing = router.build(context, width=64, height=64)

    image_kwargs = {
        "prompt": "robot, city street",
        "negative_prompt": "blurry",
        "width": 64,
        "height": 64,
        "guidance_scale": 7.5,
        "num_inference_steps": 20,
    }
    router.apply_image_branch(image_pipe, image_kwargs, routing)
    initial_result = image_pipe(**image_kwargs)
    initial_frame = router.apply_initial_frame_fallback(initial_result.images[0], routing)

    video_kwargs = {
        "image": initial_frame,
        "width": 64,
        "height": 64,
        "num_frames": 4,
        "fps": 8,
        "num_inference_steps": 25,
        "decode_chunk_size": 8,
    }
    router.apply_video_branch(video_pipe, video_kwargs, routing)
    video_result = video_pipe(**video_kwargs)
    frames = router.apply_video_postprocess(video_result.frames[0], routing)
    summary = router.summarize(routing)

    image_call = image_pipe.calls[0]
    assert image_call["control_image"] is not None
    assert image_call["controlnet_conditioning_scale"] == [0.75, 0.7]

    video_call = video_pipe.calls[0]
    assert video_call["control_image"] is not None
    assert video_call["controlnet_conditioning_scale"] == [0.75, 0.7]
    assert video_call["motion_bucket_id"] is not None
    assert video_call["noise_aug_strength"] is not None

    # Transition stitching should blend the boundary-adjacent frame.
    assert frames[1].getpixel((0, 0)) != (0, 0, 255)
    assert summary["pose_control"] is True
    assert summary["depth_control"] is True
    assert summary["video_call_summary"]["has_control_input"] is True


def test_condition_router_prefers_preprocessed_pose_and_depth_assets(tmp_path: Path) -> None:
    pose_asset = tmp_path / "pose.png"
    depth_asset = tmp_path / "depth.png"
    Image.new("RGB", (32, 32), "red").save(pose_asset)
    Image.new("RGB", (32, 32), "green").save(depth_asset)
    pose_manifest = _write_json(
        tmp_path / "pose-manifest.json",
        {
            "pose_assets": [{"frame_index": 0, "image_path": str(pose_asset)}],
            "skeleton_sequence": [],
        },
    )
    depth_manifest = _write_json(
        tmp_path / "depth-manifest.json",
        {
            "frames": [{"frame_index": 0, "depth_asset": str(depth_asset), "focus_mode": "scene"}],
        },
    )

    router = ConditionRouter()
    routing = router.build(
        GenerationContext(
            metadata={
                "pose_artifact_paths": [str(pose_manifest)],
                "depth_artifact_paths": [str(depth_manifest)],
            }
        ),
        width=32,
        height=32,
    )

    assert routing.pose_control_image is not None
    assert routing.pose_control_image.getpixel((0, 0)) == (255, 0, 0)
    assert routing.depth_control_image is not None
    assert routing.depth_control_image.getpixel((0, 0)) == (0, 128, 0)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
