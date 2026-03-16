from app.models.schemas import AdapterPlan, AdapterSignal, AdapterStepPlan, ControlPlan


class ControlSignalMapper:
    def build(self, control_plan: ControlPlan | None, model_name: str, use_mock: bool) -> AdapterPlan:
        if control_plan is None:
            return AdapterPlan(
                profile=None,
                model_family=model_name,
                execution_mode="mock" if use_mock else "direct",
                steps=[],
            )

        execution_mode = "mock" if use_mock else "adapter-ready"
        steps: list[AdapterStepPlan] = []
        for step in control_plan.steps:
            signals = [
                self._camera_signal(step),
                self._pose_signal(step),
                self._depth_signal(step),
                self._transition_signal(step),
            ]
            steps.append(
                AdapterStepPlan(
                    step_index=step.step_index,
                    start_frame=step.start_frame,
                    end_frame=step.end_frame,
                    beat=step.beat,
                    signals=signals,
                )
            )
        return AdapterPlan(
            profile=control_plan.profile,
            model_family=model_name,
            execution_mode=execution_mode,
            steps=steps,
        )

    def _camera_signal(self, step) -> AdapterSignal:
        adapter_name = "camera-path-controller"
        provider = "camera-motion-engine"
        if step.camera_path in {"static", "locked"}:
            adapter_name = "camera-static-controller"
            provider = "static-camera-engine"
        return AdapterSignal(
            adapter_type="camera",
            adapter_name=adapter_name,
            strength=step.camera_intensity,
            source=step.camera_path,
            provider=provider,
            params={
                "camera_mode": step.camera_mode,
                "path": step.camera_path,
                "intensity": step.camera_intensity,
            },
            provider_payload=self._camera_provider_payload(step, provider),
        )

    def _pose_signal(self, step) -> AdapterSignal:
        adapter_name = "openpose-adapter"
        provider = "openpose"
        if "anchor" in step.pose_hint_type:
            adapter_name = "openpose-anchor-adapter"
        return AdapterSignal(
            adapter_type="pose",
            adapter_name=adapter_name,
            strength=step.pose_hint_strength,
            source=step.pose_hint_type,
            provider=provider,
            params={
                "hint_type": step.pose_hint_type,
                "motion_label": step.motion_label,
                "strength": step.pose_hint_strength,
            },
            provider_payload=self._pose_provider_payload(step),
        )

    def _depth_signal(self, step) -> AdapterSignal:
        adapter_name = "depth-anything-adapter"
        provider = "depth-anything"
        if "midas" in step.depth_hint_type:
            adapter_name = "midas-depth-adapter"
            provider = "midas"
        elif "soft" in step.depth_hint_type:
            adapter_name = "soft-depth-adapter"
            provider = "depth-soft"
        return AdapterSignal(
            adapter_type="depth",
            adapter_name=adapter_name,
            strength=step.depth_hint_strength,
            source=step.depth_hint_type,
            provider=provider,
            params={
                "hint_type": step.depth_hint_type,
                "strength": step.depth_hint_strength,
            },
            provider_payload=self._depth_provider_payload(step, provider),
        )

    def _transition_signal(self, step) -> AdapterSignal:
        adapter_name = "transition-planner"
        provider = "segment-stitcher"
        if step.transition_type in {"cut", "hard-cut", "match-cut"}:
            adapter_name = "cut-transition-planner"
            provider = "cut-stitcher"
        return AdapterSignal(
            adapter_type="transition",
            adapter_name=adapter_name,
            strength=step.transition_strength,
            source=step.transition_type,
            provider=provider,
            params={
                "transition_type": step.transition_type,
                "strength": step.transition_strength,
            },
            provider_payload=self._transition_provider_payload(step, provider),
        )

    def _camera_provider_payload(self, step, provider: str) -> dict[str, str | int | float | bool | list[str] | list[int]]:
        return {
            "provider": provider,
            "camera_path": step.camera_path,
            "camera_mode": step.camera_mode,
            "intensity": step.camera_intensity,
            "start_frame": step.start_frame,
            "end_frame": step.end_frame,
        }

    def _pose_provider_payload(self, step) -> dict[str, str | int | float | bool | list[str] | list[int]]:
        return {
            "provider": "openpose",
            "hint_type": step.pose_hint_type,
            "strength": step.pose_hint_strength,
            "motion_label": step.motion_label,
            "frame_window": [step.start_frame, step.end_frame],
        }

    def _depth_provider_payload(self, step, provider: str) -> dict[str, str | int | float | bool | list[str] | list[int]]:
        return {
            "provider": provider,
            "hint_type": step.depth_hint_type,
            "strength": step.depth_hint_strength,
            "frame_window": [step.start_frame, step.end_frame],
            "focus_mode": "subject" if "focus" in step.depth_hint_type else "scene",
        }

    def _transition_provider_payload(self, step, provider: str) -> dict[str, str | int | float | bool | list[str] | list[int]]:
        return {
            "provider": provider,
            "transition_type": step.transition_type,
            "strength": step.transition_strength,
            "boundary_frame": step.end_frame,
            "blend_frames": max(1, round(step.transition_strength * 4)),
        }
