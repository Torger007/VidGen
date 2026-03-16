import uuid

from app.core.config import get_settings
from app.core.generation_profiles import get_generation_profile
from app.models.schemas import GenerateVideoRequest, GenerationParameters, JobRecord, utc_now
from app.services.control_plan import ControlPlanBuilder
from app.services.job_store import JobStore
from app.services.prompting import PromptOrchestrator
from app.tasks.generate_video import enqueue_generate_video


class JobService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = JobStore()
        self.prompt_orchestrator = PromptOrchestrator()
        self.control_plan_builder = ControlPlanBuilder()

    def create_job(self, request: GenerateVideoRequest) -> JobRecord:
        now = utc_now()
        profile_name, profile_defaults = get_generation_profile(
            request.generation_profile or self.settings.default_generation_profile
        )
        parameters = self._build_parameters(request, profile_defaults)
        prompt_bundle = self.prompt_orchestrator.build_bundle(
            request.prompt,
            request.style_hint,
            generation_profile=profile_name,
        )
        control_plan = self.control_plan_builder.build(prompt_bundle, profile_name, parameters.fps)
        parameters.duration_sec = max(1, round(control_plan.total_duration_sec))
        parameters.num_frames = control_plan.total_frames
        job = JobRecord(
            job_id=str(uuid.uuid4()),
            prompt=request.prompt,
            status="queued",
            reference_image_path=request.reference_image_path,
            generation_profile=profile_name,
            prompt_bundle=prompt_bundle,
            control_plan=control_plan,
            parameters=parameters,
            created_at=now,
            updated_at=now,
        )
        self.store.save(job)
        enqueue_generate_video(job.job_id)
        refreshed = self.store.load(job.job_id)
        return refreshed or job

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.store.load(job_id)

    def list_jobs(self, limit: int = 20) -> list[JobRecord]:
        return self.store.list(limit=limit)

    def _build_parameters(
        self,
        request: GenerateVideoRequest,
        profile_defaults: dict,
    ) -> GenerationParameters:
        raw_parameters = {
            "model": self.settings.default_model,
            "fps": self.settings.output_fps,
            "num_frames": self.settings.output_frames,
            "width": self.settings.output_width,
            "height": self.settings.output_height,
            "min_total_score": self.settings.default_min_total_score,
            "min_text_alignment": self.settings.default_min_text_alignment,
            "min_temporal_stability": self.settings.default_min_temporal_stability,
        }
        raw_parameters.update(profile_defaults)
        if request.parameters is not None:
            raw_parameters.update(request.parameters.model_dump(exclude_unset=True))
        parameters = GenerationParameters.model_validate(raw_parameters)
        if "duration_sec" not in raw_parameters:
            parameters.duration_sec = max(1, parameters.num_frames // max(1, parameters.fps))
        return parameters
