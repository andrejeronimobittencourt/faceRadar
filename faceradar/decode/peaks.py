"""Center / size / offset maps → face boxes (local peaks + IoU NMS)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from faceradar.config import OUTPUT_STRIDE, DecodeParams
from faceradar.io.images import Letterbox

__all__ = ['FaceBox', 'box_iou', 'extract_faces', 'non_max_suppression']


@dataclass(frozen=True)
class FaceBox:
  """One detected face: ``x1, y1, width, height`` in pixels + center confidence."""

  x1: float
  y1: float
  width: float
  height: float
  score: float

  @property
  def x2(self) -> float:
    return self.x1 + self.width

  @property
  def y2(self) -> float:
    return self.y1 + self.height

  def to_original(self, mapping: Letterbox) -> FaceBox:
    """Canvas pixels → original image pixels (inverse letterbox)."""
    return FaceBox(
      x1=(self.x1 - mapping.pad_x) / mapping.scale,
      y1=(self.y1 - mapping.pad_y) / mapping.scale,
      width=self.width / mapping.scale,
      height=self.height / mapping.scale,
      score=self.score,
    )


def box_iou(a: FaceBox, b: FaceBox) -> float:
  """Intersection over union of two boxes."""
  inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
  inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
  inter = inter_w * inter_h
  if inter <= 0.0:
    return 0.0
  union = a.width * a.height + b.width * b.height - inter
  return float(inter / union) if union > 0.0 else 0.0


def non_max_suppression(faces: list[FaceBox], iou_threshold: float) -> list[FaceBox]:
  """Greedy score-ordered NMS: drop a box overlapping a kept one by ``>= iou_threshold``."""
  kept: list[FaceBox] = []
  for face in sorted(faces, key=lambda item: item.score, reverse=True):
    if all(box_iou(face, prior) < iou_threshold for prior in kept):
      kept.append(face)
  return kept


@torch.no_grad()
def extract_faces(
  center_logits: torch.Tensor,
  size_map: torch.Tensor,
  offset_map: torch.Tensor,
  params: DecodeParams | None = None,
  stride: int = OUTPUT_STRIDE,
) -> list[FaceBox]:
  """Decode one image's heads into faces on the input canvas.

  Args:
    center_logits: ``(1, h, w)`` or ``(h, w)`` center logits.
    size_map: ``(2, h, w)`` width / height in input pixels.
    offset_map: ``(2, h, w)`` sub-cell centre offset in cells.

  Local maxima above ``conf_threshold``, then IoU NMS. Sorted by score, best first.
  """
  params = params or DecodeParams()
  center = center_logits.detach().float().cpu()
  if center.ndim == 3:
    center = center[0]
  heat = torch.sigmoid(center)[None, None]
  pooled = F.max_pool2d(heat, params.peak_kernel, stride=1, padding=params.peak_kernel // 2)
  keep = (pooled == heat) & (heat > params.conf_threshold)
  ys, xs = torch.where(keep[0, 0])
  if ys.numel() == 0:
    return []

  scores = heat[0, 0, ys, xs]
  order = torch.argsort(scores, descending=True)
  ys, xs, scores = ys[order], xs[order], scores[order]
  size = size_map.detach().float().cpu()
  offset = offset_map.detach().float().cpu()
  widths = size[0, ys, xs].clamp(min=1.0)
  heights = size[1, ys, xs].clamp(min=1.0)
  centers_x = (xs.float() + offset[0, ys, xs]) * stride
  centers_y = (ys.float() + offset[1, ys, xs]) * stride

  faces = [
    FaceBox(
      x1=float(cx - 0.5 * width),
      y1=float(cy - 0.5 * height),
      width=float(width),
      height=float(height),
      score=float(score),
    )
    for cx, cy, width, height, score in zip(
      centers_x.tolist(),
      centers_y.tolist(),
      widths.tolist(),
      heights.tolist(),
      scores.tolist(),
    )
  ]
  return non_max_suppression(faces, params.nms_iou)
