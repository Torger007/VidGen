from pathlib import Path

from PIL import Image

from app.core.config import get_settings


#负责参考图读取和尺寸处理。
class ReferenceImageService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def load(self, image_path: str, width: int, height: int) -> Image.Image:
        if not self.settings.allow_local_reference_images:
            raise ValueError("Local reference images are disabled by configuration.")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        return image.resize((width, height))
