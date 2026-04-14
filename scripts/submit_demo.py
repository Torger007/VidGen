"""
VidGen demo submission script.

Quick start (after starting the API with `uvicorn app.main:app --port 8001`):

    # Default demo case (robot-walk-city, seed=101)
    python scripts/submit_demo.py

    # Custom prompt
    python scripts/submit_demo.py --prompt "A cat sitting on a windowsill at sunset"

    # With reference image
    python scripts/submit_demo.py --prompt "A robot walking in a city" --reference-image-path storage/regression_inputs/verification-reference.png
"""
import argparse
import json
import time

import httpx

# Stable demo case defaults (Phase A verified)
DEFAULT_PROMPT = "A robot walking forward in a city street at night"
DEFAULT_MODEL = "stable-video-diffusion-img2vid"
DEFAULT_REFERENCE_IMAGE = "storage/regression_inputs/verification-reference.png"
DEFAULT_GENERATION_PROFILE = "balanced"
DEFAULT_SEED = 101


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a VidGen demo generation job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/submit_demo.py\n"
            "  python scripts/submit_demo.py --prompt 'A cat on a windowsill' --seed 42\n"
            "  python scripts/submit_demo.py --base-url http://127.0.0.1:8001\n"
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"Text prompt (default: {DEFAULT_PROMPT})")
    parser.add_argument("--style-hint", default=None)
    parser.add_argument("--reference-image-path", default=DEFAULT_REFERENCE_IMAGE, help=f"Reference image path (default: {DEFAULT_REFERENCE_IMAGE})")
    parser.add_argument("--generation-profile", default=DEFAULT_GENERATION_PROFILE, help=f"Generation profile (default: {DEFAULT_GENERATION_PROFILE})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--reference-strength", type=float, default=0.7)
    parser.add_argument("--prompt-strength", type=float, default=0.8)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--disable-control-plan", action="store_true")
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "style_hint": args.style_hint,
        "reference_image_path": args.reference_image_path,
        "generation_profile": args.generation_profile,
        "enable_control_plan": not args.disable_control_plan,
        "parameters": {
            "model": args.model,
            "duration_sec": max(1, args.num_frames // max(args.fps, 1)),
            "fps": args.fps,
            "num_frames": args.num_frames,
            "width": args.width,
            "height": args.height,
            "num_candidates": args.num_candidates,
            "retry_attempts": args.retry_attempts,
            "seed": args.seed,
            "reference_strength": args.reference_strength,
            "prompt_strength": args.prompt_strength,
        },
    }

    with httpx.Client(timeout=600.0) as client:
        created = client.post(f"{args.base_url}/v1/videos:generate", json=payload)
        created.raise_for_status()
        job = created.json()
        job_id = job["job_id"]
        print(json.dumps({"submitted": job}, ensure_ascii=False, indent=2))

        while True:
            detail = client.get(f"{args.base_url}/v1/jobs/{job_id}")
            detail.raise_for_status()
            body = detail.json()
            print(json.dumps({"job": body}, ensure_ascii=False, indent=2))
            if body["status"] in {"succeeded", "failed"}:
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
