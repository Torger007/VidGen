from app.models.schemas import CandidateSummary, JobListItem, JobRecord


def test_job_list_item_is_lighter_than_job_record() -> None:
    job_fields = set(JobRecord.model_fields)
    list_fields = set(JobListItem.model_fields)

    assert "candidates" in job_fields
    assert "candidates" not in list_fields
    assert "selected_candidate" in list_fields
    assert "selection_mode" in list_fields


def test_candidate_summary_allows_evaluation_payload() -> None:
    summary = CandidateSummary.model_validate(
        {
            "candidate_index": 1,
            "score": 0.81,
            "evaluation": {
                "total_score": 0.81,
                "metrics": {
                    "text_alignment": 0.8,
                    "temporal_stability": 0.7,
                    "motion_score": 0.6,
                    "prompt_score": 0.8,
                    "reference_score": 0.0,
                    "candidate_bias": 0.1,
                },
                "method": {
                    "text_alignment": "heuristic",
                    "temporal_stability": "frame-diff",
                    "motion_score": "frame-diff",
                },
                "diagnostics": {
                    "optical_flow_mean": None,
                    "optical_flow_std": None,
                    "frame_diff_mean": 0.1,
                    "motion_diff_mean": 0.12,
                    "sampled_frame_indices": [0, 1],
                },
                "passed_thresholds": True,
                "rejection_reasons": [],
            },
        }
    )

    assert summary.evaluation is not None
    assert summary.evaluation.metrics.text_alignment == 0.8
    assert summary.evaluation.diagnostics is not None


def test_job_record_supports_rejection_summary() -> None:
    record = JobRecord.model_validate(
        {
            "job_id": "job-1",
            "prompt": "test prompt",
            "status": "succeeded",
            "prompt_bundle": {
                "subject": "robot",
                "scene": "city",
                "style": "cinematic",
                "action": "walk",
                "camera": "slow dolly in",
                "negative_prompt": "blurry",
            },
            "parameters": {"model": "mock-svd"},
            "created_at": "2026-03-12T00:00:00Z",
            "updated_at": "2026-03-12T00:00:01Z",
            "selection_mode": "fallback-best",
            "rejection_reason_counts": {"total_score_below_threshold": 2},
        }
    )

    assert record.selection_mode == "fallback-best"
    assert record.rejection_reason_counts["total_score_below_threshold"] == 2
