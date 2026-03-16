from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.models.schemas import (
    GenerateVideoRequest,
    GenerateVideoResponse,
    HealthResponse,
    JobDetailResponse,
    JobListItem,
    ListJobsResponse,
)
from app.services.job_service import JobService


router = APIRouter()
settings = get_settings()
job_service = JobService()


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        mock_pipeline=settings.use_mock_pipeline,
    )


@router.post("/v1/videos:generate", response_model=GenerateVideoResponse, status_code=status.HTTP_202_ACCEPTED)
def generate_video(request: GenerateVideoRequest) -> GenerateVideoResponse:
    job = job_service.create_job(request)
    return GenerateVideoResponse(
        job_id=job.job_id,
        status=job.status,
        generation_profile=job.generation_profile,
        selected_candidate=job.selected_candidate,
        selection_mode=job.selection_mode,
    )


@router.get("/v1/jobs", response_model=ListJobsResponse)
def list_jobs(limit: int = 20) -> ListJobsResponse:
    items = [
        JobListItem(
            job_id=job.job_id,
            prompt=job.prompt,
            status=job.status,
            generation_profile=job.generation_profile,
            created_at=job.created_at,
            updated_at=job.updated_at,
            output_preview_path=job.output_preview_path,
            selected_candidate=job.selected_candidate,
            selection_mode=job.selection_mode,
            rejection_reason_counts=job.rejection_reason_counts,
            error_message=job.error_message,
        )
        for job in job_service.list_jobs(limit=limit)
    ]
    return ListJobsResponse(items=items)


@router.get("/v1/jobs/{job_id}", response_model=JobDetailResponse)
def get_job(job_id: str) -> JobDetailResponse:
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobDetailResponse.model_validate(job)
