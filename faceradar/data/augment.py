"""Square zoom crops for optional extra training samples.

A crop is always inside the image (no padding). Two modes:

- **face crop**: centred on one random face; side length is the face side times a
  log-uniform factor in ``face_context``
- **patch crop**: a random square of ``patch_side`` × the image short side

Faces whose centre falls outside the crop are dropped from that sample; faces cut
by the border are clipped. The dataset appends these as extra indices — it does
not replace the full-image samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = ['ZoomCropConfig', 'crop_boxes', 'zoom_crop']


@dataclass(frozen=True)
class ZoomCropConfig:
  """Zoom-crop ranges.

  ``face_share`` is the fraction of crops centred on a face (images with no faces
  always use a patch crop).
  """

  face_share: float = 0.5
  face_context: tuple[float, float] = (1.25, 4.0)
  patch_side: tuple[float, float] = (0.3, 1.0)

  def __post_init__(self) -> None:
    if not 0.0 <= self.face_share <= 1.0:
      raise ValueError(f'face_share must be in [0, 1], got {self.face_share}')
    low, high = self.face_context
    if not 1.0 <= low <= high:
      raise ValueError(f'face_context must satisfy 1 <= low <= high, got {self.face_context}')
    low, high = self.patch_side
    if not 0.0 < low <= high <= 1.0:
      raise ValueError(f'patch_side must satisfy 0 < low <= high <= 1, got {self.patch_side}')


def crop_boxes(boxes: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
  """Move ``x1, y1, w, h`` boxes into a ``side``-pixel square crop at ``(x0, y0)``.

  Keeps boxes whose centre is inside the crop, clipped to its border.
  """
  boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
  x1 = boxes[:, 0] - x0
  y1 = boxes[:, 1] - y0
  x2 = x1 + boxes[:, 2]
  y2 = y1 + boxes[:, 3]
  cx = 0.5 * (x1 + x2)
  cy = 0.5 * (y1 + y2)
  inside = (cx >= 0.0) & (cx < side) & (cy >= 0.0) & (cy < side)
  x1, y1, x2, y2 = (np.clip(edge[inside], 0.0, side) for edge in (x1, y1, x2, y2))
  out = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)
  return out[(out[:, 2] > 0.0) & (out[:, 3] > 0.0)]


def _face_crop_origin(
  box: np.ndarray,
  side: int,
  width: int,
  height: int,
  rng: np.random.Generator,
) -> tuple[int, int]:
  """Random crop origin keeping ``box`` inside the crop and the crop inside the image."""
  origin = []
  for start, extent, limit in ((box[0], box[2], width), (box[1], box[3], height)):
    low = max(0.0, start + extent - side)
    high = min(start, limit - side)
    if low <= high:
      value = rng.uniform(low, high)
    else:
      value = min(max(start + 0.5 * extent - 0.5 * side, 0.0), limit - side)
    origin.append(int(round(value)))
  return origin[0], origin[1]


def zoom_crop(
  image: Image.Image,
  boxes: np.ndarray,
  rng: np.random.Generator,
  config: ZoomCropConfig,
) -> tuple[Image.Image, np.ndarray]:
  """Return a square zoom crop of ``image`` / ``boxes`` (see module docs)."""
  width, height = image.size
  short = min(width, height)
  if short < 2:
    return image, boxes
  boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
  valid = boxes[(boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)]

  if len(valid) and rng.random() < config.face_share:
    box = valid[int(rng.integers(len(valid)))]
    low, high = config.face_context
    factor = math.exp(rng.uniform(math.log(low), math.log(high)))
    side = int(round(min(max(float(box[2]), float(box[3]), 1.0) * factor, short)))
    side = max(side, 1)
    x0, y0 = _face_crop_origin(box, side, width, height, rng)
  else:
    low, high = config.patch_side
    side = max(1, int(round(rng.uniform(low, high) * short)))
    x0 = int(rng.integers(0, width - side + 1))
    y0 = int(rng.integers(0, height - side + 1))

  cropped = image.crop((x0, y0, x0 + side, y0 + side))
  return cropped, crop_boxes(boxes, x0, y0, side)
