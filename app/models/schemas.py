from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "succeeded", "failed"]


#放请求相应和内部数据结构定义
class PromptBundle(BaseModel):
    subject: str
    scene: str
    scene_template: str
    style: str
    action: str
    camera: str
    shot_plan: list["ShotPlanStep"] = Field(default_factory=list)
    negative_prompt: str


class ShotPlanStep(BaseModel):
    beat: str
    camera: str
    motion: str
    duration_sec: float = Field(default=1.0, ge=0.2, le=8.0)
    emphasis: str


class ControlPlanStep(BaseModel):
    step_index: int
    beat: str
    camera_mode: str
    camera_path: str
    camera_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    motion_label: str
    motion_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    pose_hint_type: str
    pose_hint_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    depth_hint_type: str
    depth_hint_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    transition_type: str
    transition_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    duration_sec: float = Field(default=1.0, ge=0.2, le=8.0)
    frame_count: int = Field(default=8, ge=1, le=240)
    start_frame: int = Field(default=0, ge=0)
    end_frame: int = Field(default=0, ge=0)
    emphasis: str


class ControlPlan(BaseModel):
    profile: str | None = None
    total_duration_sec: float = Field(default=3.0, ge=0.2, le=16.0)
    total_frames: int = Field(default=24, ge=1, le=240)
    dominant_camera: str
    dominant_camera_path: str | None = None
    motion_labels: list[str] = Field(default_factory=list)
    pose_hint_types: list[str] = Field(default_factory=list)
    depth_hint_types: list[str] = Field(default_factory=list)
    transition_types: list[str] = Field(default_factory=list)
    steps: list[ControlPlanStep] = Field(default_factory=list)


class AdapterSignal(BaseModel):
    adapter_type: str
    adapter_name: str
    enabled: bool = True
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    provider: str | None = None
    provider_payload: dict[str, str | int | float | bool | list[str] | list[int]] = Field(
        default_factory=dict
    )


class AdapterStepPlan(BaseModel):
    step_index: int
    start_frame: int
    end_frame: int
    beat: str
    signals: list[AdapterSignal] = Field(default_factory=list)


class AdapterPlan(BaseModel):
    profile: str | None = None
    model_family: str | None = None
    execution_mode: str
    steps: list[AdapterStepPlan] = Field(default_factory=list)


class ProviderArtifact(BaseModel):
    provider: str
    adapter_type: str
    artifact_type: str
    step_index: int
    signal_index: int
    artifact_path: str
    summary: str


class ProviderExecutionStep(BaseModel):
    step_index: int
    beat: str
    artifacts: list[ProviderArtifact] = Field(default_factory=list)


class ProviderExecutionPlan(BaseModel):
    job_id: str
    execution_mode: str
    root_dir: str
    steps: list[ProviderExecutionStep] = Field(default_factory=list)
    artifact_index: dict[str, list[str]] = Field(default_factory=dict)


class GenerationContext(BaseModel):
    prompt_suffixes: list[str] = Field(default_factory=list)
    pose_guidance: str | None = None
    depth_guidance: str | None = None
    transition_guidance: str | None = None
    camera_guidance: str | None = None
    decode_chunk_size: int = 8
    frame_budget_adjustment: int = 0
    metadata: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class GenerationParameters(BaseModel):
    model: str = "mock-svd"
    duration_sec: int = Field(default=3, ge=1, le=8)
    fps: int = Field(default=8, ge=4, le=24)
    num_frames: int = Field(default=24, ge=8, le=96)
    width: int = Field(default=576, ge=256, le=1280)
    height: int = Field(default=320, ge=256, le=1280)
    seed: int | None = None
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_candidates: int = Field(default=1, ge=1, le=4)
    retry_attempts: int = Field(default=1, ge=1, le=3)
    reference_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    prompt_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    min_total_score: float = Field(default=0.45, ge=0.0, le=1.0)
    min_text_alignment: float = Field(default=0.35, ge=0.0, le=1.0)
    min_temporal_stability: float = Field(default=0.3, ge=0.0, le=1.0)


class GenerateVideoRequest(BaseModel):
    prompt: str = Field(min_length=8, max_length=1000)
    style_hint: str | None = Field(default=None, max_length=200)
    reference_image_path: str | None = None
    generation_profile: str | None = None
    parameters: GenerationParameters | None = None


class EvaluationMetrics(BaseModel):
    text_alignment: float
    temporal_stability: float
    motion_score: float
    prompt_score: float
    reference_score: float
    candidate_bias: float


class EvaluationDiagnostics(BaseModel):
    optical_flow_mean: float | None = None
    optical_flow_std: float | None = None
    frame_diff_mean: float | None = None
    motion_diff_mean: float | None = None
    sampled_frame_indices: list[int] = Field(default_factory=list)


class EvaluationMethod(BaseModel):
    text_alignment: str
    temporal_stability: str
    motion_score: str


class EvaluationSummary(BaseModel):
    total_score: float
    metrics: EvaluationMetrics
    method: EvaluationMethod
    diagnostics: EvaluationDiagnostics | None = None
    passed_thresholds: bool = True
    rejection_reasons: list[str] = Field(default_factory=list)


class CandidateSummary(BaseModel):
    candidate_index: int
    score: float
    seed: int | None = None
    attempts_used: int | None = None
    path: str | None = None
    preview_path: str | None = None
    video_path: str | None = None
    evaluation: EvaluationSummary | None = None


class JobRecord(BaseModel):
    job_id: str
    prompt: str
    status: JobStatus
    reference_image_path: str | None = None
    generation_profile: str | None = None
    prompt_bundle: PromptBundle
    control_plan: ControlPlan | None = None
    parameters: GenerationParameters
    created_at: datetime
    updated_at: datetime
    output_path: str | None = None
    output_preview_path: str | None = None
    candidate_paths: list[str] = Field(default_factory=list)
    selected_candidate: CandidateSummary | None = None
    candidates: list[CandidateSummary] = Field(default_factory=list)
    selection_mode: str | None = None
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    error_message: str | None = None


class GenerateVideoResponse(BaseModel):
    job_id: str
    status: JobStatus
    generation_profile: str | None = None
    selected_candidate: CandidateSummary | None = None
    selection_mode: str | None = None


class JobDetailResponse(JobRecord):
    pass


class JobListItem(BaseModel):
    job_id: str
    prompt: str
    status: JobStatus
    generation_profile: str | None = None
    created_at: datetime
    updated_at: datetime
    output_preview_path: str | None = None
    selected_candidate: CandidateSummary | None = None
    selection_mode: str | None = None
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    error_message: str | None = None


class ListJobsResponse(BaseModel):
    items: list[JobListItem]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    mock_pipeline: bool


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


for _model in (PromptBundle,):
    rebuild = getattr(_model, "model_rebuild", None)
    if callable(rebuild):
        rebuild()
    else:  # pragma: no cover - pydantic v1 compatibility
        _model.update_forward_refs()
