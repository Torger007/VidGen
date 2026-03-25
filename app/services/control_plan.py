from app.models.schemas import ControlPlan, ControlPlanStep, PromptBundle


#负责shot plan生成逐段控制计划
class ControlPlanBuilder:
    def build(self, prompt_bundle: PromptBundle, generation_profile: str | None, fps: int) -> ControlPlan:
        steps: list[ControlPlanStep] = []
        current_start = 0
        motion_labels: list[str] = []
        pose_hint_types: list[str] = []
        depth_hint_types: list[str] = []
        transition_types: list[str] = []
        for index, step in enumerate(prompt_bundle.shot_plan, start=1):
            frame_count = max(1, round(step.duration_sec * fps))
            end_frame = current_start + frame_count - 1
            steps.append(
                ControlPlanStep(
                    step_index=index,
                    beat=step.beat,
                    camera_mode=step.camera,
                    camera_path=self._camera_path(step.motion, step.camera),
                    camera_intensity=self._camera_intensity(step.motion, generation_profile),
                    motion_label=step.motion,
                    motion_strength=self._motion_strength(step.motion, generation_profile),
                    pose_hint_type=self._pose_hint_type(step.motion, generation_profile),
                    pose_hint_strength=self._pose_hint_strength(step.motion, generation_profile),
                    depth_hint_type=self._depth_hint_type(prompt_bundle.scene_template, generation_profile),
                    depth_hint_strength=self._depth_hint_strength(prompt_bundle.scene_template, generation_profile),
                    transition_type=self._transition_type(index, len(prompt_bundle.shot_plan), generation_profile),
                    transition_strength=self._transition_strength(index, generation_profile),
                    duration_sec=step.duration_sec,
                    frame_count=frame_count,
                    start_frame=current_start,
                    end_frame=end_frame,
                    emphasis=step.emphasis,
                )
            )
            current_start = end_frame + 1
            motion_labels.append(step.motion)
            pose_hint_types.append(steps[-1].pose_hint_type)
            depth_hint_types.append(steps[-1].depth_hint_type)
            transition_types.append(steps[-1].transition_type)

        total_duration = round(sum(step.duration_sec for step in prompt_bundle.shot_plan), 2) or 1.0
        total_frames = max(1, sum(step.frame_count for step in steps))
        dominant_camera = steps[0].camera_mode if steps else prompt_bundle.camera
        return ControlPlan(
            profile=generation_profile,
            total_duration_sec=total_duration,
            total_frames=total_frames,
            dominant_camera=dominant_camera,
            dominant_camera_path=steps[0].camera_path if steps else "static",
            motion_labels=motion_labels,
            pose_hint_types=pose_hint_types,
            depth_hint_types=depth_hint_types,
            transition_types=transition_types,
            steps=steps,
        )

    def _camera_path(self, motion: str, camera: str) -> str:
        lowered_motion = motion.lower()
        lowered_camera = camera.lower()
        if "arc" in lowered_motion:
            return "arc"
        if "push" in lowered_motion or "push" in lowered_camera:
            return "push-in"
        if "track" in lowered_motion or "tracking" in lowered_motion:
            return "track"
        if "settle" in lowered_motion or "stop" in lowered_motion:
            return "settle"
        if "locked" in lowered_camera or "minimal" in lowered_motion:
            return "static"
        return "guided"

    def _camera_intensity(self, motion: str, profile: str | None) -> float:
        base = 0.45
        if profile == "strict":
            base -= 0.15
        if profile == "creative":
            base += 0.2
        if any(token in motion.lower() for token in ["energetic", "sweeping", "aggressive"]):
            base += 0.15
        return max(0.1, min(1.0, base))

    def _motion_strength(self, motion: str, profile: str | None) -> float:
        base = 0.5
        if profile == "strict":
            base -= 0.1
        if profile == "creative":
            base += 0.2
        if "gentle" in motion.lower() or "minimal" in motion.lower():
            base -= 0.1
        if "energetic" in motion.lower() or "sweeping" in motion.lower():
            base += 0.1
        return max(0.1, min(1.0, base))

    def _pose_hint_type(self, motion: str, profile: str | None) -> str:
        lowered = motion.lower()
        if "run" in lowered or "energetic" in lowered:
            return "openpose-dynamic"
        if "walk" in lowered or "track" in lowered:
            return "openpose-track"
        if profile == "strict":
            return "openpose-anchor"
        return "openpose-sparse"

    def _pose_hint_strength(self, motion: str, profile: str | None) -> float:
        base = 0.45 if profile != "strict" else 0.6
        if profile == "creative":
            base = 0.4
        if "dynamic" in motion.lower() or "energetic" in motion.lower():
            base += 0.1
        return max(0.1, min(1.0, base))

    def _depth_hint_type(self, scene_template: str, profile: str | None) -> str:
        if scene_template in {"city", "forest"}:
            return "depth-anything-v2"
        if profile == "strict":
            return "midas-structured"
        return "depth-soft"

    def _depth_hint_strength(self, scene_template: str, profile: str | None) -> float:
        base = 0.5
        if scene_template in {"city", "forest"}:
            base += 0.1
        if profile == "strict":
            base += 0.1
        if profile == "creative":
            base -= 0.05
        return max(0.1, min(1.0, base))

    def _transition_type(self, index: int, total_steps: int, profile: str | None) -> str:
        if index == 1:
            return "cold-open"
        if index == total_steps:
            return "ease-out" if profile != "creative" else "flare-fade"
        if profile == "strict":
            return "match-cut"
        if profile == "creative":
            return "motion-blend"
        return "cut"

    def _transition_strength(self, index: int, profile: str | None) -> float:
        base = 0.3
        if index == 1:
            base = 0.2
        if profile == "creative":
            base += 0.25
        if profile == "strict":
            base -= 0.05
        return max(0.1, min(1.0, base))
