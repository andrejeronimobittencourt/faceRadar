"""Single-image inference pipeline (letterbox, predict, peak decode, map back)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from faceradar.config import DecodeParams, Paths
from faceradar.decode.peaks import FaceBox, extract_faces
from faceradar.io.images import letterbox, load_rgb, plane_to_original
from faceradar.training.checkpoints import LoadedModel, load_model
from faceradar.training.loop import resolve_device
from faceradar.viz.inference import render_detection_panels


def load_model_from_path(model_path: str | Path, device: str = 'auto') -> LoadedModel:
  """Load a trained ``.pt`` checkpoint (``train.py``) for inference."""
  return load_model(model_path, resolve_device(device))


@torch.no_grad()
def predict_on_image(
  loaded: LoadedModel,
  image: str | Path | Image.Image,
  decode_params: DecodeParams | None = None,
) -> dict:
  """Detect every face in one image.

  The image is letterboxed to the model input exactly as in training; all
  outputs are mapped back onto the original image. Returns:

  - ``image``: ``(H, W, 3)`` uint8 original RGB
  - ``faces``: ``list[FaceBox]`` in original pixels, best score first
  - ``center_heatmap``: ``(H, W)`` face-centre confidence in ``[0, 1]``
  - ``size_map``: ``(H, W)`` predicted face side ``√(w·h)`` in original pixels
  """
  original = image.convert('RGB') if isinstance(image, Image.Image) else load_rgb(image)
  canvas, mapping = letterbox(original)
  rgb = np.asarray(canvas, dtype=np.float32) / 255.0
  batch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(loaded.device)
  output = loaded.model(batch)

  faces = [
    face.to_original(mapping)
    for face in extract_faces(
      output['center'][0],
      output['size'][0],
      output['offset'][0],
      decode_params,
    )
  ]
  center = torch.sigmoid(output['center'][0, 0].float()).cpu().numpy()
  size = output['size'][0].float().clamp(min=0.0).cpu().numpy()
  side = np.sqrt(size[0] * size[1]) / mapping.scale
  return {
    'image': np.asarray(original),
    'faces': faces,
    'center_heatmap': plane_to_original(center, mapping, original.size),
    'size_map': plane_to_original(side, mapping, original.size),
  }


def run_infer_cli(
  image_path: str | Path,
  model_path: str | Path | None = None,
  output_path: str | Path | None = None,
  device: str = 'auto',
  decode_params: DecodeParams | None = None,
  show_scores: bool = True,
) -> dict:
  """CLI helper: load model, detect, print faces, show or save the 3-panel figure."""
  import matplotlib.pyplot as plt

  decode_params = decode_params or DecodeParams()
  model_path = Path(model_path or Paths().face_radar)
  loaded = load_model_from_path(model_path, device)
  print(
    f'Model loaded: {model_path} (stacks={loaded.meta.get("stacks")}, '
    f'epoch={loaded.meta.get("epoch")})'
  )
  result = predict_on_image(loaded, image_path, decode_params)
  faces: list[FaceBox] = result['faces']
  print(f'{len(faces)} faces (conf >= {decode_params.conf_threshold})')
  for face in faces:
    print(f'  conf={face.score:.2f}  {face.width:.0f}×{face.height:.0f}px')
  fig = render_detection_panels(
    result['image'],
    faces,
    result['center_heatmap'],
    result['size_map'],
    show_scores=show_scores,
  )

  if output_path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
    print(f'Saved plot to {output_path}')
    plt.close(fig)
  else:
    plt.show()

  return result
