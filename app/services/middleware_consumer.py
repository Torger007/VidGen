import json
from pathlib import Path

from app.models.schemas import GenerationContext, ProviderExecutionPlan


class MiddlewareConsumer:
    def build_context(self, execution: ProviderExecutionPlan) -> GenerationContext:
        context = GenerationContext()
        artifact_index = execution.artifact_index

        skeleton_paths = artifact_index.get("skeleton_sequence", [])
        if skeleton_paths:
            context.metadata["pose_artifact_paths"] = skeleton_paths
        if skeleton_paths:
            skeleton_info = self._load_json(Path(skeleton_paths[0]))
            context.pose_guidance = "pose-conditioned motion continuity"
            context.prompt_suffixes.append("consistent body pose across frames")
            context.metadata["pose_artifacts"] = len(skeleton_paths)
            pose_assets = skeleton_info.get("pose_assets", [])
            context.metadata["pose_frames"] = max(
                len(skeleton_info.get("skeleton_sequence", [])),
                len(pose_assets),
            )
            if pose_assets:
                context.metadata["pose_asset_images"] = [
                    str(item.get("image_path"))
                    for item in pose_assets
                    if isinstance(item, dict) and item.get("image_path")
                ]

        depth_paths = artifact_index.get("depth_manifest", [])
        if depth_paths:
            context.metadata["depth_artifact_paths"] = depth_paths
        if depth_paths:
            depth_info = self._load_json(Path(depth_paths[0]))
            focus_mode = "scene"
            frames = depth_info.get("frames", [])
            if frames:
                focus_mode = frames[0].get("focus_mode", "scene")
            context.depth_guidance = f"depth-aware composition ({focus_mode})"
            context.prompt_suffixes.append(f"{focus_mode} depth structure preserved")
            context.metadata["depth_artifacts"] = len(depth_paths)
            context.metadata["depth_focus_mode"] = focus_mode
            context.metadata["depth_asset_images"] = [
                str(frame.get("depth_asset"))
                for frame in frames
                if isinstance(frame, dict) and frame.get("depth_asset")
            ]

        transition_paths = artifact_index.get("transition_manifest", [])
        if transition_paths:
            context.metadata["transition_artifact_paths"] = transition_paths
        if transition_paths:
            transition_info = self._load_json(Path(transition_paths[0]))
            manifest = transition_info.get("manifest", {})
            blend_frames = int(manifest.get("blend_frames", 1))
            context.transition_guidance = f"smooth segment transition with {blend_frames} blend frames"
            context.decode_chunk_size = max(4, min(16, 8 + blend_frames))
            context.metadata["transition_artifacts"] = len(transition_paths)
            context.metadata["transition_blend_frames"] = blend_frames

        camera_paths = artifact_index.get("camera_manifest", [])
        if camera_paths:
            context.metadata["camera_artifact_paths"] = camera_paths
        if camera_paths:
            camera_info = self._load_json(Path(camera_paths[0]))
            trajectory = camera_info.get("trajectory", {})
            camera_path = trajectory.get("path", "guided")
            context.camera_guidance = f"camera path follows {camera_path}"
            context.prompt_suffixes.append(f"camera motion: {camera_path}")
            context.metadata["camera_artifacts"] = len(camera_paths)
            context.metadata["camera_path"] = camera_path

        context.frame_budget_adjustment = min(
            8,
            len(skeleton_paths) + len(depth_paths) + len(transition_paths),
        )
        return context

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
