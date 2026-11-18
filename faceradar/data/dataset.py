"""Face-box records → (letterboxed image, center / size / offset targets) samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from faceradar.config import HEATMAP_SIZE, INPUT_SIZE, OUTPUT_STRIDE
from faceradar.data.augment import ZoomCropConfig, zoom_crop
from faceradar.data.csv_format import FaceRecord, resolve_image_path
from faceradar.data.targets import boxes_to_targets
from faceradar.io.images import boxes_to_canvas, letterbox

Sample = dict[str, torch.Tensor]


def seed_worker(worker_id: int) -> None:
  """DataLoader ``worker_init_fn``: give each worker its own augmentation RNG.

  Forked workers otherwise inherit an identical ``np.random.Generator`` state
  and draw the same augmentation sequence.
  """
  info = torch.utils.data.get_worker_info()
  if info is None:
    return
  if isinstance(info.dataset, FaceBoxDataset):
    info.dataset.reseed(info.seed)


def mirror_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
  """Horizontally mirror ``x1, y1, w, h`` boxes inside an image ``width`` pixels wide."""
  out = boxes.copy()
  out[:, 0] = width - (boxes[:, 0] + boxes[:, 2])
  return out


class FaceBoxDataset(Dataset):
  """One ``FaceRecord`` → ``(3, 512, 512)`` image + center / size / offset / mask targets.

  Letterboxed like inference (aspect kept, bilinear, black padding), RGB in
  ``[0, 1]``. Indices ``0 .. N-1`` are the full image. With ``zoom_extra > 0`` and
  ``augment``, ``int(round(N * zoom_extra))`` close-up crops are appended (see
  ``faceradar.data.augment``). Training augmentation also includes horizontal flip
  (p = ``flip_prob``) and brightness / contrast jitter ×[0.7, 1.3] (p = 0.5 each).
  """

  def __init__(
    self,
    records: list[FaceRecord],
    image_dir: str | Path,
    input_size: int = INPUT_SIZE,
    heatmap_size: int = HEATMAP_SIZE,
    stride: int = OUTPUT_STRIDE,
    min_face_side: float = 4.0,
    augment: bool = False,
    flip_prob: float = 0.5,
    zoom: ZoomCropConfig | None = None,
    zoom_extra: float = 0.0,
    seed: int = 0,
  ) -> None:
    if not 0.0 <= zoom_extra:
      raise ValueError(f'zoom_extra must be >= 0, got {zoom_extra}')
    self.records = records
    self.image_dir = Path(image_dir)
    self.input_size = int(input_size)
    self.heatmap_size = int(heatmap_size)
    self.stride = int(stride)
    self.min_face_side = float(min_face_side)
    self.augment = bool(augment)
    self.flip_prob = float(flip_prob)
    self.zoom = zoom or ZoomCropConfig()
    self.zoom_extra = float(zoom_extra) if augment else 0.0
    self._base_len = len(records)
    self._extra_len = int(round(self._base_len * self.zoom_extra))
    self._rng = np.random.default_rng(seed)

  def __len__(self) -> int:
    return self._base_len + self._extra_len

  def reseed(self, seed: int) -> None:
    """Replace the augmentation RNG (called once per DataLoader worker)."""
    self._rng = np.random.default_rng(seed % (2**32))

  def __getitem__(self, index: int) -> Sample:
    use_zoom = index >= self._base_len
    record = self.records[index % self._base_len]
    image = Image.open(resolve_image_path(self.image_dir, record.image)).convert('RGB')
    boxes = np.asarray(record.boxes, dtype=np.float32).reshape(-1, 4)
    if use_zoom:
      image, boxes = zoom_crop(image, boxes, self._rng, self.zoom)
    if self.augment:
      image, boxes = self._photometric(image, boxes)

    canvas, mapping = letterbox(image, self.input_size)
    targets = boxes_to_targets(
      boxes_to_canvas(boxes, mapping),
      input_size=self.input_size,
      heatmap_size=self.heatmap_size,
      stride=self.stride,
      min_face_side=self.min_face_side,
    )
    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    return {
      'image': torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
      **{key: torch.from_numpy(value) for key, value in targets.items()},
    }

  def _photometric(
    self,
    image: Image.Image,
    boxes: np.ndarray,
  ) -> tuple[Image.Image, np.ndarray]:
    """Flip + photometric jitter; boxes move with the pixels."""
    if self._rng.random() < self.flip_prob:
      boxes = mirror_boxes(boxes, image.size[0])
      image = ImageOps.mirror(image)
    if self._rng.random() < 0.5:
      image = ImageEnhance.Brightness(image).enhance(float(self._rng.uniform(0.7, 1.3)))
    if self._rng.random() < 0.5:
      image = ImageEnhance.Contrast(image).enhance(float(self._rng.uniform(0.7, 1.3)))
    return image, boxes
