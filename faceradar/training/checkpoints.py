"""Checkpoint format, compatibility checks, resume state and model loading.

Two files per run:

- ``<name>.pt``: best weights by validation loss + metadata (for inference).
- ``<name>.last.pt``: full training state after every epoch (weights,
  optimizer, LR scheduler, AMP scaler, epoch, best loss); resumed automatically.

Metadata records everything needed to rebuild and decode the model, so
inference never has to guess stacks, width, input size or stride.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from faceradar.config import HEATMAP_SIZE, INPUT_SIZE, OUTPUT_STRIDE
from faceradar.data.augment import ZoomCropConfig
from faceradar.models.hourglass import FaceRadarNet, build_face_radar

CHECKPOINT_FORMAT_VERSION = 1
# Keys that must match for weights / training state to be reused.
COMPATIBILITY_KEYS = (
  'stacks',
  'filters',
  'input_size',
  'heatmap_size',
  'output_stride',
)
# Geometry baked into decode / letterboxing; a checkpoint must agree with it.
GEOMETRY = {
  'input_size': INPUT_SIZE,
  'heatmap_size': HEATMAP_SIZE,
  'output_stride': OUTPUT_STRIDE,
}


def checkpoint_meta(
  *,
  stacks: int,
  filters: int,
  intermediate_supervision: bool,
  zoom: ZoomCropConfig,
  zoom_extra: float,
) -> dict[str, object]:
  """Metadata stored in every checkpoint of a run.

  Architecture / geometry keys gate reuse (``COMPATIBILITY_KEYS``); the training
  recipe is recorded for reference only.
  """
  return {
    'format_version': CHECKPOINT_FORMAT_VERSION,
    'stacks': int(stacks),
    'filters': int(filters),
    **GEOMETRY,
    'intermediate_supervision': bool(intermediate_supervision),
    'zoom_extra': float(zoom_extra),
    'zoom_crop': {
      'face_share': float(zoom.face_share),
      'face_context': list(zoom.face_context),
      'patch_side': list(zoom.patch_side),
    },
  }


def last_checkpoint_path(path: Path) -> Path:
  """Full-state checkpoint written every epoch next to ``path``."""
  path = Path(path)
  return path.with_name(f'{path.stem}.last{path.suffix}')


def assert_compatible(path: Path, checkpoint: dict, meta: dict[str, object]) -> None:
  """Refuse to reuse a checkpoint written for another architecture / geometry.

  Raises:
    RuntimeError: Lists every mismatching key as ``(found, expected)``.
  """
  mismatched = {
    key: (checkpoint.get(key), meta[key])
    for key in COMPATIBILITY_KEYS
    if key in meta and checkpoint.get(key) != meta[key]
  }
  if mismatched:
    raise RuntimeError(
      f'{path} is incompatible with this run {mismatched} (found, expected). '
      'Pass --fresh to start over, or --output to write elsewhere.'
    )


def save_checkpoint(path: Path, payload: dict) -> None:
  """Write atomically so an interrupted save never corrupts the previous file."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_name(path.name + '.tmp')
  torch.save(payload, tmp_path)
  os.replace(tmp_path, path)


def read_resume_state(
  output_path: Path,
  meta: dict[str, object],
  device: torch.device,
  fresh: bool,
) -> dict | None:
  """Return the ``.last`` training state to resume from, or None to start fresh.

  Raises:
    RuntimeError: Existing checkpoints are incompatible, or a best checkpoint
      exists without resumable state (pass ``--fresh`` to overwrite).
  """
  if fresh:
    return None
  output_path = Path(output_path)
  last_path = last_checkpoint_path(output_path)
  if output_path.is_file():
    assert_compatible(
      output_path,
      torch.load(output_path, map_location='cpu', weights_only=False),
      meta,
    )
  if last_path.is_file():
    state = torch.load(last_path, map_location=device, weights_only=False)
    assert_compatible(last_path, state, meta)
    return state
  if output_path.is_file():
    raise RuntimeError(
      f'{output_path} exists but has no resumable state ({last_path.name}). '
      'Pass --fresh to overwrite it.'
    )
  return None


@dataclass(frozen=True)
class LoadedModel:
  """Model in eval mode + the metadata it was trained with."""

  model: FaceRadarNet
  meta: dict
  device: torch.device


def load_model(path: str | Path, device: torch.device) -> LoadedModel:
  """Rebuild a trained detector from a ``.pt`` checkpoint.

  Raises:
    FileNotFoundError: Missing checkpoint.
    RuntimeError: Not a training checkpoint, or trained for another input size /
      stride than this package decodes.
  """
  path = Path(path)
  if not path.is_file():
    raise FileNotFoundError(f'Model checkpoint not found: {path}')
  checkpoint = torch.load(path, map_location=device, weights_only=False)
  if 'model' not in checkpoint:
    raise RuntimeError(f'{path} is not a training checkpoint (no "model" weights)')
  mismatched = {
    key: (checkpoint[key], expected)
    for key, expected in GEOMETRY.items()
    if key in checkpoint and checkpoint[key] != expected
  }
  if mismatched:
    raise RuntimeError(f'{path} was trained with other geometry {mismatched} (found, expected)')
  filters = checkpoint.get('filters')
  model = build_face_radar(
    int(checkpoint.get('stacks', 2)),
    None if filters is None else int(filters),
  ).to(device)
  model.load_state_dict(checkpoint['model'])
  model.eval()
  meta = {key: value for key, value in checkpoint.items() if key != 'model'}
  return LoadedModel(model=model, meta=meta, device=device)
