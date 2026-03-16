import argparse
import json
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a VidGen demo generation job.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--style-hint", default=None)
    parser.add_argument("--reference-image-path", default=None)
    parser.add_argument("--generation-profile", default=None)
    parser.add_argument("--model", default="mock-svd")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--reference-strength", type=float, default=0.7)
    parser.add_argument("--prompt-strength", type=float, default=0.8)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "style_hint": args.style_hint,
        "reference_image_path": args.reference_image_path,
        "generation_profile": args.generation_profile,
        "parameters": {
            "model": args.model,
            "duration_sec": max(1, args.num_frames // max(args.fps, 1)),
            "fps": args.fps,
            "num_frames": args.num_frames,
            "width": args.width,
            "height": args.height,
            "num_candidates": args.num_candidates,
            "retry_attempts": args.retry_attempts,
            "reference_strength": args.reference_strength,
            "prompt_strength": args.prompt_strength,
        },
    }

    with httpx.Client(timeout=60.0) as client:
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
