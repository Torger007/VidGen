from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path


faulthandler.enable()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal local SDXL load/inference check.")
    parser.add_argument(
        "--model-dir",
        default=r"b:\agent\MyCode\VidGen\storage\models\stable-diffusion-xl-base-1.0",
        help="Local diffusers directory for Stable Diffusion XL.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device to use.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Generated image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Generated image height.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Inference steps for the smoke test.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Guidance scale for the smoke test.",
    )
    parser.add_argument(
        "--prompt",
        default="A robot walking in a city street at night, cinematic lighting",
        help="Prompt for the smoke test.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, distorted, low detail",
        help="Negative prompt for the smoke test.",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Only load the pipeline without running inference.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir)

    print(f"MODEL_DIR={model_dir}")
    print(f"MODEL_EXISTS={model_dir.exists()}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    import torch
    from diffusers import StableDiffusionXLPipeline

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"TORCH_VERSION={torch.__version__}")
    print(f"DEVICE={args.device}")
    print(f"DTYPE={dtype}")
    print("LOADING_PIPELINE=1")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = pipe.to(args.device)

    print("PIPELINE_LOADED=1")
    print(f"PIPELINE_CLASS={pipe.__class__.__name__}")

    if args.skip_infer:
        return 0

    print("RUNNING_INFERENCE=1")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    image = result.images[0]
    print("INFERENCE_OK=1")
    print(f"IMAGE_SIZE={image.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
