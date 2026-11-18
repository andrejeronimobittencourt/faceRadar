"""Face-box CSV format: parsing, validation and image path resolution.

One row per face::

  filename,x1,y1,w,h

- ``filename``: image path. Resolved as given (absolute, or relative to the image
  directory), then by basename under the image directory.
- ``x1``, ``y1``: top-left corner of the face box in original image pixels.
- ``w``, ``h``: box width / height in pixels.
- Rows of the same ``filename`` are grouped into one training image (order of
  first appearance is kept).
- A row with ``w`` or ``h`` equal to ``0`` lists an image with no faces.
- Any other columns are ignored.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_COLUMN = 'filename'
BOX_COLUMNS = ('x1', 'y1', 'w', 'h')


class FaceCsvError(ValueError):
  """The face-box CSV does not follow the expected format."""


@dataclass(frozen=True)
class FaceRecord:
  """One image reference + ``(N, 4)`` float32 boxes as ``x1, y1, w, h`` in pixels."""

  image: str
  boxes: np.ndarray


def validate_header(fieldnames: list[str] | None, csv_path: Path) -> None:
  """Require the ``filename`` column and every box column.

  Raises:
    FaceCsvError: Missing columns.
  """
  present = set(fieldnames or [])
  missing = [column for column in (IMAGE_COLUMN, *BOX_COLUMNS) if column not in present]
  if missing:
    raise FaceCsvError(
      f'{csv_path}: missing required column(s): {", ".join(missing)}. '
      f'Expected {IMAGE_COLUMN},{",".join(BOX_COLUMNS)}.'
    )


def _parse_value(raw: str | None, csv_path: Path, line: int, column: str) -> float:
  if raw is None or not raw.strip():
    raise FaceCsvError(f'{csv_path}:{line}: empty {column} cell')
  try:
    value = float(raw)
  except ValueError as error:
    raise FaceCsvError(f'{csv_path}:{line}: column {column} is not a number: {raw!r}') from error
  if not math.isfinite(value):
    raise FaceCsvError(f'{csv_path}:{line}: column {column} is not finite: {raw!r}')
  return value


def parse_row(row: dict[str, str], csv_path: Path, line: int) -> tuple[str, np.ndarray | None]:
  """Parse one CSV row into ``(image, box)``; ``box`` is None for a no-face row.

  Raises:
    FaceCsvError: Empty image cell, non-numeric value or negative size.
  """
  image = (row.get(IMAGE_COLUMN) or '').strip()
  if not image:
    raise FaceCsvError(f'{csv_path}:{line}: empty {IMAGE_COLUMN} cell')
  x1, y1, width, height = (
    _parse_value(row.get(column), csv_path, line, column) for column in BOX_COLUMNS
  )
  if width < 0.0 or height < 0.0:
    raise FaceCsvError(f'{csv_path}:{line}: negative box size w={width} h={height}')
  if width == 0.0 or height == 0.0:
    return image, None
  return image, np.asarray([x1, y1, width, height], dtype=np.float32)


def load_face_csv(csv_path: str | Path) -> list[FaceRecord]:
  """Load and validate a face-box CSV, grouped per image.

  Raises:
    FileNotFoundError: Missing CSV.
    FaceCsvError: Malformed header / values, or no rows.
  """
  csv_path = Path(csv_path)
  if not csv_path.is_file():
    raise FileNotFoundError(f'Face CSV not found: {csv_path}')
  grouped: dict[str, list[np.ndarray]] = {}
  with csv_path.open('r', newline='', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    validate_header(reader.fieldnames, csv_path)
    for line, row in enumerate(reader, start=2):
      image, box = parse_row(row, csv_path, line)
      boxes = grouped.setdefault(image, [])
      if box is not None:
        boxes.append(box)
  if not grouped:
    raise FaceCsvError(f'{csv_path}: no rows')
  return [
    FaceRecord(
      image=image,
      boxes=np.stack(boxes) if boxes else np.zeros((0, 4), dtype=np.float32),
    )
    for image, boxes in grouped.items()
  ]


def resolve_image_path(image_dir: str | Path, image: str) -> Path:
  """Locate an image referenced by the ``filename`` column.

  Order: ``filename`` as given if absolute, ``image_dir / filename``, then
  ``image_dir / basename(filename)``, then ``filename`` relative to the working
  directory.

  Raises:
    FileNotFoundError: None of the candidates exist.
  """
  image_dir = Path(image_dir)
  reference = Path(image)
  candidates = [image_dir / reference, image_dir / reference.name, reference]
  if reference.is_absolute():
    candidates.insert(0, reference)
  for candidate in candidates:
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(f'Image {image!r} not found (looked under {image_dir})')
