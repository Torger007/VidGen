from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path


faulthandler.enable()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal local SVD load/inference check.")
    parser.add_argument(
        "--model-dir",
        default=r"b:\agent\MyCode\VidGen\storage\models\stable-video-diffusion-img2vid-xt",
        help="Local diffusers directory for Stable Video Diffusion.",
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
        help="Input image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Input image height.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames to generate for the smoke test.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="FPS metadata for the generated sample.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Inference steps for the smoke test.",
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=2,
        help="Decode chunk size for the smoke test.",
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
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"TORCH_VERSION={torch.__version__}")
    print(f"DEVICE={args.device}")
    print(f"DTYPE={dtype}")
    print("LOADING_PIPELINE=1")

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = pipe.to(args.device)

    print("PIPELINE_LOADED=1")
    print(f"PIPELINE_CLASS={pipe.__class__.__name__}")

    if args.skip_infer:
        return 0

    image = Image.new("RGB", (args.width, args.height), color=(200, 200, 200))
    print("RUNNING_INFERENCE=1")
    result = pipe(
        image=image,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.steps,
        decode_chunk_size=args.decode_chunk_size,
    )
    frames = result.frames[0]
    print(f"INFERENCE_OK=1")
    print(f"FRAME_COUNT={len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
