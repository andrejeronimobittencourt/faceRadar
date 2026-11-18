"""Shared configuration and defaults."""

from dataclasses import dataclass
from pathlib import Path

INPUT_SIZE = 512
OUTPUT_STRIDE = 4
HEATMAP_SIZE = INPUT_SIZE // OUTPUT_STRIDE

__all__ = [
  'HEATMAP_SIZE',
  'INPUT_SIZE',
  'OUTPUT_STRIDE',
  'DecodeParams',
  'Paths',
  'TrainingConfig',
]


@dataclass(frozen=True)
class Paths:
  """Default filesystem paths (all relative to repo root)."""

  models_dir: Path = Path('models')
  assets_dir: Path = Path('assets')
  data_dir: Path = Path('data')
  train_csv: Path = Path('data') / 'train.csv'
  train_images: Path = Path('data') / 'train'
  val_csv: Path = Path('data') / 'val.csv'
  val_images: Path = Path('data') / 'val'
  face_radar: Path = Path('models') / 'face_radar.pt'


@dataclass(frozen=True)
class TrainingConfig:
  """Training hyperparameters (CLI defaults for ``train.py``)."""

  epochs: int = 40
  batch_size: int = 8
  learning_rate: float = 1e-3
  min_learning_rate: float = 1e-6
  stacks: int = 2
  filters: int = 64
  center_weight: float = 1.0
  size_weight: float = 0.1
  offset_weight: float = 1.0
  min_face_side: float = 4.0
  flip_prob: float = 0.5
  zoom_extra: float = 0.5
  zoom_face_share: float = 0.5
  workers: int = 4
  seed: int = 42


@dataclass(frozen=True)
class DecodeParams:
  """Peak decode settings: confidence floor, local-max kernel, IoU NMS, cap."""

  conf_threshold: float = 0.3
  peak_kernel: int = 3
  nms_iou: float = 0.4
  max_faces: int = 200

  def __post_init__(self) -> None:
    if not 0.0 < self.conf_threshold < 1.0:
      raise ValueError(f'conf_threshold must be in (0, 1), got {self.conf_threshold}')
    if self.peak_kernel < 1 or self.peak_kernel % 2 == 0:
      raise ValueError(f'peak_kernel must be a positive odd number, got {self.peak_kernel}')
    if not 0.0 < self.nms_iou <= 1.0:
      raise ValueError(f'nms_iou must be in (0, 1], got {self.nms_iou}')
    if self.max_faces < 1:
      raise ValueError(f'max_faces must be >= 1, got {self.max_faces}')
