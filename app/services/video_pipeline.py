import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

from app.core.config import get_settings
from app.core.model_registry import get_model_spec
from app.models.schemas import (
    AdapterPlan,
    ControlPlan,
    GenerationContext,
    GenerationParameters,
    JobRecord,
    PromptBundle,
    ProviderExecutionPlan,
)
from app.services.adapter_executor import AdapterExecutor
from app.services.artifacts import ArtifactWriter
from app.services.condition_router import ConditionRouter
from app.services.control_signal_mapper import ControlSignalMapper
from app.services.diffusers_loader import (
    DiffusersUnavailableError,
    build_generation_context_key,
    load_pipelines,
)
from app.services.middleware_consumer import MiddlewareConsumer
from app.services.reference_images import ReferenceImageService
from app.services.scoring import CandidateScorer


#核心
class VideoPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.writer = ArtifactWriter()
        self.reference_images = ReferenceImageService()
        self.scorer = CandidateScorer()
        self.signal_mapper = ControlSignalMapper()
        self.adapter_executor = AdapterExecutor()
        self.middleware_consumer = MiddlewareConsumer()
        self.condition_router = ConditionRouter()

    def render(self, job: JobRecord) -> str:
        if self.settings.use_mock_pipeline:
            return self._render_mock(job)
        return self._render_open_source(
            job.prompt_bundle,
            job.control_plan,
            job.parameters,
            job.job_id,
            reference_image_path=job.reference_image_path,
        )

    #模拟生产流程
    def _render_mock(self, job: JobRecord) -> str:
        output_dir = self.settings.storage_root / "outputs"
        output_path = output_dir / f"{job.job_id}.json"
        preview_path = output_dir / f"{job.job_id}.png"
        candidates_dir = self.writer.ensure_dir(output_dir / job.job_id)
        selected_seed = job.parameters.seed if job.parameters.seed is not None else random.randint(1, 999999)
        candidates = []
        for index in range(job.parameters.num_candidates):
            candidate_path = candidates_dir / f"candidate-{index + 1}.json"
            candidate_preview_path = candidates_dir / f"candidate-{index + 1}.png"
            evaluation = self.scorer.evaluate(
                prompt_bundle=job.prompt_bundle,
                parameters=job.parameters,
                used_reference_image=job.reference_image_path is not None,
                candidate_index=index,
            )
            candidate_payload = {
                "candidate_index": index + 1,
                "score": evaluation["total_score"],
                "evaluation": evaluation,
                "seed": selected_seed + index,
                "reference_image_path": job.reference_image_path,
                "frames": [
                    {
                        "index": frame_index,
                        "caption": f"Frame {frame_index}: {job.prompt_bundle.action}",
                        "camera": job.prompt_bundle.camera,
                    }
                    for frame_index in range(job.parameters.num_frames)
                ],
            }
            self.writer.write_preview_image(
                candidate_preview_path,
                prompt=f"Candidate {index + 1}\n{job.prompt_bundle.action}",
                width=job.parameters.width,
                height=job.parameters.height,
            )
            self.writer.write_json(candidate_path, candidate_payload)
            candidates.append(
                {
                    "candidate_index": index + 1,
                    "score": evaluation["total_score"],
                    "evaluation": evaluation,
                    "path": str(candidate_path),
                    "preview_path": str(candidate_preview_path),
                    "seed": selected_seed + index,
                }
            )
        passed_candidates = [item for item in candidates if item["evaluation"]["passed_thresholds"]]
        best_candidate = max(passed_candidates or candidates, key=lambda item: item["score"])
        adapter_plan = self._build_adapter_plan(job.control_plan, job.parameters.model, use_mock=True)
        provider_execution = self._execute_adapter_plan(
            job.job_id,
            adapter_plan,
            reference_image_path=job.reference_image_path,
            frame_size=(job.parameters.width, job.parameters.height),
        )
        generation_context = self._build_generation_context(provider_execution)
        payload = {
            "job_id": job.job_id,
            "mode": "mock",
            "reference_image_path": job.reference_image_path,
            "prompt_bundle": job.prompt_bundle.model_dump(),
            "control_plan": job.control_plan.model_dump() if job.control_plan else None,
            "adapter_plan": adapter_plan,
            "provider_execution": provider_execution,
            "generation_context": generation_context,
            "parameters": job.parameters.model_dump(),
            "seed": selected_seed,
            "selected_candidate": best_candidate,
            "candidates": candidates,
            "selection_mode": "thresholded-best" if passed_candidates else "fallback-best",
            "rejection_reason_counts": self._collect_rejection_reason_counts(candidates),
        }
        self.writer.write_preview_image(
            preview_path,
            prompt=(
                f"{job.prompt_bundle.subject}\n"
                f"{job.prompt_bundle.action}\n"
                f"{self._context_prompt_line(generation_context)}"
            ),
            width=job.parameters.width,
            height=job.parameters.height,
        )
        payload["preview_path"] = str(preview_path)
        self.writer.write_json(output_path, payload)
        return str(output_path)

    def _render_open_source(
        self,
        prompt_bundle: PromptBundle,
        control_plan: ControlPlan | None,
        parameters: GenerationParameters,
        job_id: str,
        reference_image_path: str | None = None,
    ) -> str:
        spec = get_model_spec(parameters.model)
        if spec.provider != "diffusers":
            raise ValueError(f"Model provider '{spec.provider}' is not implemented.")

        output_dir = self.settings.storage_root / "outputs"
        preview_path = output_dir / f"{job_id}.png"
        video_path = output_dir / f"{job_id}.mp4"
        metadata_path = output_dir / f"{job_id}.json"
        candidates_dir = self.writer.ensure_dir(output_dir / job_id)
        adapter_plan = self._build_adapter_plan(control_plan, spec.name, use_mock=False)
        provider_execution = self._execute_adapter_plan(
            job_id,
            adapter_plan,
            reference_image_path=reference_image_path,
            frame_size=(parameters.width, parameters.height),
        )
        generation_context = self._build_generation_context(provider_execution)

        try:
            pipelines = load_pipelines(
                spec.name,
                build_generation_context_key(GenerationContext.model_validate(generation_context)),
                skip_image_pipeline=reference_image_path is not None,
            )
            candidates = []
            for index in range(parameters.num_candidates):
                candidate = self._render_candidate(
                    image_pipe=pipelines["image"],
                    video_pipe=pipelines["video"],
                    prompt_bundle=prompt_bundle,
                    parameters=parameters,
                    spec_name=spec.name,
                    image_steps=spec.default_image_steps,
                    video_steps=spec.default_num_inference_steps,
                    output_dir=candidates_dir,
                    candidate_index=index,
                    reference_image_path=reference_image_path,
                    generation_context=GenerationContext.model_validate(generation_context),
                )
                candidates.append(candidate)
        except DiffusersUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Open-source video generation failed: {exc}") from exc

        passed_candidates = [item for item in candidates if item["evaluation"]["passed_thresholds"]]
        best_candidate = max(passed_candidates or candidates, key=lambda item: item["score"])
        selected_preview = Path(best_candidate["preview_path"])
        selected_video = Path(best_candidate["video_path"])
        self.writer.write_image(preview_path, Image.open(selected_preview).convert("RGB"))
        video_path.write_bytes(selected_video.read_bytes())

        payload = {
            "job_id": job_id,
            "mode": "diffusers",
            "model": spec.name,
            "model_id": spec.model_id,
            "reference_image_path": reference_image_path,
            "prompt_bundle": prompt_bundle.model_dump(),
            "control_plan": control_plan.model_dump() if control_plan else None,
            "adapter_plan": adapter_plan,
            "provider_execution": provider_execution,
            "generation_context": generation_context,
            "parameters": parameters.model_dump(),
            "preview_path": str(preview_path),
            "video_path": str(video_path),
            "selected_candidate": best_candidate,
            "candidates": candidates,
            "selection_mode": "thresholded-best" if passed_candidates else "fallback-best",
            "rejection_reason_counts": self._collect_rejection_reason_counts(candidates),
        }
        return self.writer.write_json(metadata_path, payload)

    def _render_candidate(
        self,
        *,
        image_pipe: Any,
        video_pipe: Any,
        prompt_bundle: PromptBundle,
        parameters: GenerationParameters,
        spec_name: str,
        image_steps: int,
        video_steps: int,
        output_dir: Path,
        candidate_index: int,
        reference_image_path: str | None,
        generation_context: GenerationContext | None,
    ) -> dict[str, Any]:
        last_error: str | None = None
        candidate_seed = (parameters.seed if parameters.seed is not None else random.randint(1, 999999)) + candidate_index
        for attempt in range(parameters.retry_attempts):
            try:
                initial_frame = self._resolve_initial_frame(
                    image_pipe=image_pipe,
                    prompt_bundle=prompt_bundle,
                    parameters=parameters,
                    image_steps=image_steps,
                    reference_image_path=reference_image_path,
                    seed=candidate_seed + attempt,
                    generation_context=generation_context,
                )
                frames = self._generate_video_frames(
                    video_pipe,
                    initial_frame,
                    parameters,
                    video_steps,
                    model_name=spec_name,
                    generation_context=generation_context,
                )
                routing_summary = self._build_routing_summary(
                    generation_context=generation_context,
                    parameters=parameters,
                    image_pipe=image_pipe,
                    video_pipe=video_pipe,
                )
                preview_path = output_dir / f"candidate-{candidate_index + 1}.png"
                video_path = output_dir / f"candidate-{candidate_index + 1}.mp4"
                self.writer.write_image(preview_path, initial_frame)
                self.writer.write_video(video_path, frames, fps=parameters.fps)
                evaluation = self.scorer.evaluate(
                    prompt_bundle=prompt_bundle,
                    parameters=parameters,
                    used_reference_image=reference_image_path is not None,
                    candidate_index=candidate_index,
                    initial_frame=initial_frame,
                    frames=frames,
                )
                return {
                    "candidate_index": candidate_index + 1,
                    "score": evaluation["total_score"],
                    "evaluation": evaluation,
                    "preview_path": str(preview_path),
                    "video_path": str(video_path),
                    "seed": candidate_seed + attempt,
                    "attempts_used": attempt + 1,
                    "routing_summary": routing_summary,
                }
            except Exception as exc:
                last_error = str(exc)
        raise RuntimeError(
            f"Candidate {candidate_index + 1} failed after {parameters.retry_attempts} attempts: {last_error}"
        )

    def _resolve_initial_frame(
        self,
        image_pipe: Any,
        prompt_bundle: PromptBundle,
        parameters: GenerationParameters,
        image_steps: int,
        reference_image_path: str | None,
        seed: int | None = None,
        generation_context: GenerationContext | None = None,
    ) -> Image.Image:
        if reference_image_path:
            reference = self.reference_images.load(reference_image_path, parameters.width, parameters.height)
            return self._blend_reference_with_prompt(
                reference_image=reference,
                image_pipe=image_pipe,
                prompt_bundle=prompt_bundle,
                parameters=parameters,
                image_steps=image_steps,
                seed=seed,
                generation_context=generation_context,
            )
        return self._generate_initial_frame(
            image_pipe,
            prompt_bundle,
            parameters,
            image_steps,
            seed,
            generation_context=generation_context,
        )

    def _generate_initial_frame(
        self,
        image_pipe: Any,
        prompt_bundle: PromptBundle,
        parameters: GenerationParameters,
        image_steps: int,
        seed: int | None = None,
        generation_context: GenerationContext | None = None,
    ) -> Image.Image:
        prompt = self._compose_positive_prompt(prompt_bundle, generation_context)
        routing = self.condition_router.build(
            generation_context,
            width=parameters.width,
            height=parameters.height,
        )
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": prompt_bundle.negative_prompt,
            "width": parameters.width,
            "height": parameters.height,
            "guidance_scale": parameters.guidance_scale,
            "num_inference_steps": image_steps,
        }
        self.condition_router.apply_image_branch(image_pipe, call_kwargs, routing)
        generator = self._build_generator(seed)
        if generator is not None:
            call_kwargs["generator"] = generator
        result = image_pipe(
            **call_kwargs,
        )
        return self.condition_router.apply_initial_frame_fallback(result.images[0], routing)

    def _blend_reference_with_prompt(
        self,
        *,
        reference_image: Image.Image,
        image_pipe: Any,
        prompt_bundle: PromptBundle,
        parameters: GenerationParameters,
        image_steps: int,
        seed: int | None,
        generation_context: GenerationContext | None = None,
    ) -> Image.Image:
        if parameters.reference_strength >= 0.99:
            return reference_image
        if image_pipe is None:
            return reference_image
        generated = self._generate_initial_frame(
            image_pipe=image_pipe,
            prompt_bundle=prompt_bundle,
            parameters=parameters,
            image_steps=image_steps,
            seed=seed,
            generation_context=generation_context,
        )
        reference = reference_image.resize(generated.size).convert("RGB")
        generated = generated.convert("RGB")
        blended = Image.blend(generated, reference, parameters.reference_strength)
        enhancer = ImageEnhance.Contrast(blended)
        return enhancer.enhance(max(0.8, parameters.prompt_strength + 0.2))


    #视频帧生成逻辑
    def _generate_video_frames(
        self,
        video_pipe: Any,
        initial_frame: Image.Image,
        parameters: GenerationParameters,
        default_steps: int,
        model_name: str,
        generation_context: GenerationContext | None = None,
    ) -> list[Image.Image]:
        #1.获取解码块大小
        decode_chunk_size = generation_context.decode_chunk_size if generation_context else 8

        #2.动态调整帧数
        effective_num_frames = parameters.num_frames + (
            generation_context.frame_budget_adjustment if generation_context else 0
        )

        #3.构建路由信息
        routing = self.condition_router.build(
            generation_context,
            width=parameters.width,
            height=parameters.height,
        )

        #4. 根据模型类型调用
        if model_name.startswith("stable-video-diffusion"):
            call_kwargs = {
                "image": initial_frame,
                "width": parameters.width,
                "height": parameters.height,
                "num_frames": effective_num_frames,
                "fps": parameters.fps,
                "num_inference_steps": default_steps,
                "decode_chunk_size": decode_chunk_size,
            }
            #5.注入视频侧控制参数
            self.condition_router.apply_video_branch(video_pipe, call_kwargs, routing)

            #6.执行生成
            result = video_pipe(**call_kwargs)
        else:
            raise ValueError(f"Video generation is not implemented for model '{model_name}'.")

        #7.统一格式化为 PIL Image
        frames = result.frames[0]
        normalized = [frame if isinstance(frame, Image.Image) else Image.fromarray(frame) for frame in frames]

        #8.后处理
        return self.condition_router.apply_video_postprocess(normalized, routing)

    def _build_generator(self, seed: int | None) -> Any:
        if seed is None:
            return None
        try:
            import torch
        except ImportError:
            return None
        device = "cuda" if self.settings.device == "cuda" else "cpu"
        return torch.Generator(device=device).manual_seed(seed)

    def _collect_rejection_reason_counts(self, candidates: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in candidates:
            evaluation = candidate.get("evaluation") or {}
            for reason in evaluation.get("rejection_reasons", []):
                counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _build_adapter_plan(
        self,
        control_plan: ControlPlan | None,
        model_name: str,
        *,
        use_mock: bool,
    ) -> dict[str, Any]:
        adapter_plan: AdapterPlan = self.signal_mapper.build(control_plan, model_name, use_mock)
        return adapter_plan.model_dump()

    def _execute_adapter_plan(
        self,
        job_id: str,
        adapter_plan_payload: dict[str, Any],
        *,
        reference_image_path: str | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        adapter_plan = AdapterPlan.model_validate(adapter_plan_payload)
        execution = self.adapter_executor.execute(
            job_id=job_id,
            adapter_plan=adapter_plan,
            root_dir=self.settings.adapter_artifact_root,
            source_image_path=reference_image_path,
            frame_size=frame_size,
        )
        return execution.model_dump()


    def _compose_positive_prompt(
        self,
        prompt_bundle: PromptBundle,
        generation_context: GenerationContext | None = None,
    ) -> str:
        parts = [
            prompt_bundle.subject,
            prompt_bundle.scene,
            prompt_bundle.style,
            prompt_bundle.action,
            prompt_bundle.camera,
        ]
        if generation_context:
            if generation_context.pose_guidance:
                parts.append(generation_context.pose_guidance)
            if generation_context.depth_guidance:
                parts.append(generation_context.depth_guidance)
            if generation_context.transition_guidance:
                parts.append(generation_context.transition_guidance)
            if generation_context.camera_guidance:
                parts.append(generation_context.camera_guidance)
            parts.extend(generation_context.prompt_suffixes)
        return ", ".join(parts)

    def _build_generation_context(self, provider_execution_payload: dict[str, Any]) -> dict[str, Any]:
        execution = ProviderExecutionPlan.model_validate(provider_execution_payload)
        context = self.middleware_consumer.build_context(execution)
        return context.model_dump()

    def _build_routing_summary(
        self,
        *,
        generation_context: GenerationContext | None,
        parameters: GenerationParameters,
        image_pipe: Any,
        video_pipe: Any,
    ) -> dict[str, Any]:
        routing = self.condition_router.build(
            generation_context,
            width=parameters.width,
            height=parameters.height,
        )
        image_kwargs: dict[str, Any] = {
            "width": parameters.width,
            "height": parameters.height,
            "guidance_scale": parameters.guidance_scale,
        }
        if image_pipe is not None:
            self.condition_router.apply_image_branch(image_pipe, image_kwargs, routing)
        video_kwargs: dict[str, Any] = {
            "width": parameters.width,
            "height": parameters.height,
            "num_frames": parameters.num_frames + (
                generation_context.frame_budget_adjustment if generation_context else 0
            ),
            "fps": parameters.fps,
            "decode_chunk_size": generation_context.decode_chunk_size if generation_context else 8,
        }
        self.condition_router.apply_video_branch(video_pipe, video_kwargs, routing)
        return self.condition_router.summarize(routing)

    def _context_prompt_line(self, generation_context: dict[str, Any]) -> str:
        suffixes = generation_context.get("prompt_suffixes", [])
        return suffixes[0] if suffixes else "base generation context"
