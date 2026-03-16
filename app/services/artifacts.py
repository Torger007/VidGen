import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


class ArtifactWriter:
    def write_json(self, path: Path, payload: dict[str, Any]) -> str:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    def write_preview_image(self, path: Path, prompt: str, width: int, height: int) -> str:
        image = Image.new("RGB", (width, height), color=(18, 22, 32))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), prompt[:180], fill=(235, 235, 235))
        image.save(path)
        return str(path)

    def write_image(self, path: Path, image: Image.Image) -> str:
        image.save(path)
        return str(path)

    def write_video(self, path: Path, frames: list[Image.Image], fps: int) -> str:
        array_frames = [np.asarray(frame.convert("RGB")) for frame in frames]
        imageio.mimsave(path, array_frames, fps=fps)
        return str(path)

    def ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path
