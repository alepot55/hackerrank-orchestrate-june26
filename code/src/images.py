"""Image loading, cost-aware downscaling, and base64 encoding.

Downscaling the long edge before upload is the single biggest lever on image
token cost. We re-encode to JPEG at a moderate quality which keeps surface
damage (scratches, dents, cracks) readable while cutting tokens substantially.
Encoded results are memoised in-process so multi-image claims and re-runs within
one process don't re-encode.
"""
from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from . import config


def image_id_from_path(rel_path: str) -> str:
    """'images/test/case_001/img_1.jpg' -> 'img_1'."""
    return Path(rel_path.strip()).stem


def split_image_paths(image_paths_field: str) -> List[str]:
    """Split the semicolon-separated image_paths field into clean relative paths."""
    return [p.strip() for p in image_paths_field.split(";") if p.strip()]


@lru_cache(maxsize=512)
def _encode(abs_path: str, max_edge: int, quality: int) -> Tuple[str, str]:
    """Return (media_type, base64_data) for a downscaled JPEG copy of the image."""
    with Image.open(abs_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / float(longest)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return "image/jpeg", data


def encode_image(rel_path: str) -> dict | None:
    """Build an Anthropic image content block for one relative image path.

    Returns None if the file is missing (the claim is still processed; the model
    simply sees fewer images).
    """
    abs_path = (config.IMAGES_DIR / rel_path.strip()).resolve()
    if not abs_path.exists():
        return None
    media_type, data = _encode(str(abs_path), config.IMAGE_MAX_EDGE, config.IMAGE_JPEG_QUALITY)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }
