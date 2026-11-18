"""Letterboxing between original images and the square model canvas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from faceradar.config import INPUT_SIZE

__all__ = [
  'Letterbox',
  'boxes_to_canvas',
  'letterbox',
  'load_rgb',
  'plane_to_original',
]


@dataclass(frozen=True)
class Letterbox:
  """Mapping original pixels → canvas pixels: ``canvas = original * scale + pad``."""

  scale: float
  pad_x: int
  pad_y: int
  content_width: int
  content_height: int
  size: int


def load_rgb(path: str | Path) -> Image.Image:
  """Open an image as RGB.

  Raises:
    FileNotFoundError: Missing file.
  """
  path = Path(path)
  if not path.is_file():
    raise FileNotFoundError(f'Image not found: {path}')
  return Image.open(path).convert('RGB')


def letterbox(image: Image.Image, size: int = INPUT_SIZE) -> tuple[Image.Image, Letterbox]:
  """Resize keeping the aspect ratio (bilinear) and centre on a black ``size×size`` canvas."""
  width, height = image.size
  scale = min(size / max(width, 1), size / max(height, 1))
  content_width = max(1, int(round(width * scale)))
  content_height = max(1, int(round(height * scale)))
  resized = image.resize((content_width, content_height), Image.Resampling.BILINEAR)
  canvas = Image.new('RGB', (size, size), (0, 0, 0))
  pad_x = (size - content_width) // 2
  pad_y = (size - content_height) // 2
  canvas.paste(resized, (pad_x, pad_y))
  return canvas, Letterbox(
    scale=float(scale),
    pad_x=pad_x,
    pad_y=pad_y,
    content_width=content_width,
    content_height=content_height,
    size=size,
  )


def boxes_to_canvas(boxes: np.ndarray, mapping: Letterbox) -> np.ndarray:
  """``(N, 4)`` original-pixel ``x1, y1, w, h`` → canvas pixels."""
  boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
  out = boxes.copy()
  out[:, 0] = boxes[:, 0] * mapping.scale + mapping.pad_x
  out[:, 1] = boxes[:, 1] * mapping.scale + mapping.pad_y
  out[:, 2:] = boxes[:, 2:] * mapping.scale
  return out


def plane_to_original(
  plane: np.ndarray,
  mapping: Letterbox,
  original_size: tuple[int, int],
) -> np.ndarray:
  """Heatmap-resolution ``(h, w)`` plane → ``(H, W)`` over the original image.

  Upsamples to the canvas, drops the letterbox padding, then resizes the content
  to ``original_size`` (``(width, height)``), all bilinear.
  """
  canvas = Image.fromarray(np.asarray(plane, dtype=np.float32), mode='F').resize(
    (mapping.size, mapping.size),
    Image.Resampling.BILINEAR,
  )
  content = canvas.crop(
    (
      mapping.pad_x,
      mapping.pad_y,
      mapping.pad_x + mapping.content_width,
      mapping.pad_y + mapping.content_height,
    ),
  )
  return np.asarray(content.resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32)
