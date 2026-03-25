from app.core.prompt_profiles import get_prompt_profile
from app.models.schemas import PromptBundle, ShotPlanStep


#将原始prompt转换为结构化prompt
class PromptOrchestrator:
    def build_bundle(
        self,
        prompt: str,
        style_hint: str | None = None,
        generation_profile: str | None = None,
    ) -> PromptBundle:
        _, profile = get_prompt_profile(generation_profile)
        normalized = " ".join(prompt.strip().split())
        scene, scene_template = self._infer_scene(normalized, profile)
        camera = self._infer_camera(normalized, profile)
        action = self._infer_action(normalized, profile)
        style = self._merge_style(style_hint, profile["style_base"])
        shot_plan = self._build_shot_plan(profile, action)
        return PromptBundle(
            subject=self._slice(normalized, 80),
            scene=scene,
            scene_template=scene_template,
            style=style,
            action=action,
            camera=camera,
            shot_plan=shot_plan,
            negative_prompt=profile["negative_prompt"],
        )

    def _infer_scene(self, prompt: str, profile: dict) -> tuple[str, str]:
        lowered = prompt.lower()
        if "city" in lowered:
            template = "city"
            base = profile["scene_city"]
        elif "forest" in lowered:
            template = "forest"
            base = profile["scene_forest"]
        elif "room" in lowered or "interior" in lowered:
            template = "interior"
            base = profile["scene_interior"]
        else:
            template = "default"
            base = profile["scene_default"]
        return f"{base}, {profile['scene_suffix']}", template

    def _infer_action(self, prompt: str, profile: dict) -> str:
        lowered = prompt.lower()
        if "run" in lowered:
            return profile["action_run"]
        if "walk" in lowered:
            return profile["action_walk"]
        return profile["action_default"]

    def _infer_camera(self, prompt: str, profile: dict) -> str:
        lowered = prompt.lower()
        if "close-up" in lowered:
            return profile["camera_closeup"]
        if "drone" in lowered:
            return profile["camera_drone"]
        return profile["camera_default"]

    def _merge_style(self, style_hint: str | None, base_style: str) -> str:
        if style_hint:
            return f"{base_style}, {style_hint}"
        return base_style

    def _build_shot_plan(self, profile: dict, action: str) -> list[ShotPlanStep]:
        return [
            ShotPlanStep(
                beat=step["beat"],
                camera=step["camera"],
                motion=step["motion"],
                duration_sec=step["duration_sec"],
                emphasis=f"{step['emphasis']}; action focus: {action}",
            )
            for step in profile["shot_plan"]
        ]

    def _slice(self, value: str, max_len: int) -> str:
        return value if len(value) <= max_len else value[: max_len - 3] + "..."
