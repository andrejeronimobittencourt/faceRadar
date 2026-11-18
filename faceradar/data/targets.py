"""Face boxes → center heatmap, size / offset maps and the center-cell mask."""

from __future__ import annotations

import math

import numpy as np

from faceradar.config import HEATMAP_SIZE, INPUT_SIZE, OUTPUT_STRIDE

__all__ = ['boxes_to_targets', 'draw_gaussian', 'gaussian_radius']


def gaussian_radius(width: float, height: float, min_overlap: float = 0.7) -> float:
  """CenterNet radius: a box whose corners stay inside it keeps ``min_overlap`` IoU."""
  width = max(float(width), 1.0)
  height = max(float(height), 1.0)

  a1 = 1.0
  b1 = height + width
  c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
  r1 = 0.5 * (b1 + math.sqrt(max(b1 * b1 - 4.0 * a1 * c1, 0.0)))

  a2 = 4.0
  b2 = 2.0 * (height + width)
  c2 = (1.0 - min_overlap) * width * height
  r2 = 0.5 * (b2 + math.sqrt(max(b2 * b2 - 4.0 * a2 * c2, 0.0)))

  a3 = 4.0 * min_overlap
  b3 = -2.0 * min_overlap * (height + width)
  c3 = (min_overlap - 1.0) * width * height
  r3 = 0.5 * (b3 + math.sqrt(max(b3 * b3 - 4.0 * a3 * c3, 0.0)))

  return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center_x: float, center_y: float, radius: float) -> None:
  """Max-blend a 2D Gaussian (peak exactly 1.0) at cell ``(int(x), int(y))``, in place."""
  half = max(int(2 * radius + 1), 1) // 2
  sigma = max(radius / 3.0, 0.5)
  height, width = heatmap.shape
  x = int(center_x)
  y = int(center_y)
  left = min(x, half)
  right = min(width - x - 1, half)
  top = min(y, half)
  bottom = min(height - y - 1, half)
  if left < 0 or right < 0 or top < 0 or bottom < 0:
    return
  gy = np.arange(-top, bottom + 1, dtype=np.float64)[:, None]
  gx = np.arange(-left, right + 1, dtype=np.float64)[None, :]
  patch = np.exp(-(gx * gx + gy * gy) / (2.0 * sigma * sigma)).astype(np.float32)
  region = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
  np.maximum(region, patch, out=region)


def boxes_to_targets(
  boxes: np.ndarray,
  input_size: int = INPUT_SIZE,
  heatmap_size: int = HEATMAP_SIZE,
  stride: int = OUTPUT_STRIDE,
  min_face_side: float = 4.0,
) -> dict[str, np.ndarray]:
  """Build training targets from boxes on the letterboxed input canvas.

  Args:
    boxes: ``(N, 4)`` as ``x1, y1, w, h`` in input pixels.
    min_face_side: Faces narrower or shorter than this (input pixels) get no target.

  Returns:
    ``center`` ``(1, h, w)`` Gaussian peaks; ``size`` ``(2, h, w)`` face width /
    height in input pixels at each centre cell; ``offset`` ``(2, h, w)`` residual
    from the integer cell to the true centre, in cells; ``mask`` ``(h, w)`` 1 at
    centre cells (size / offset are supervised only there).
  """
  center = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
  size = np.zeros((2, heatmap_size, heatmap_size), dtype=np.float32)
  offset = np.zeros((2, heatmap_size, heatmap_size), dtype=np.float32)
  mask = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

  for x1, y1, width, height in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
    if width < min_face_side or height < min_face_side:
      continue
    cx = x1 + 0.5 * width
    cy = y1 + 0.5 * height
    if not (0.0 <= cx < input_size and 0.0 <= cy < input_size):
      continue
    cx_cell = cx / stride
    cy_cell = cy / stride
    ix = int(cx_cell)
    iy = int(cy_cell)
    if not (0 <= ix < heatmap_size and 0 <= iy < heatmap_size):
      continue
    radius = max(gaussian_radius(width / stride, height / stride), 1.0)
    draw_gaussian(center, cx_cell, cy_cell, radius)
    size[0, iy, ix] = width
    size[1, iy, ix] = height
    offset[0, iy, ix] = cx_cell - ix
    offset[1, iy, ix] = cy_cell - iy
    mask[iy, ix] = 1.0

  return {'center': center[None], 'size': size, 'offset': offset, 'mask': mask}
