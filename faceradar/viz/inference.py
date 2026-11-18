"""Three-panel figure: detections, center heatmap, size map (all over the photo)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from faceradar.decode.peaks import FaceBox

PANEL_HEIGHT_IN = 4.2
TITLE_HEIGHT_IN = 0.4
PANEL_GAP_IN = 0.05
FIGURE_DPI = 120
BOX_COLOR = '#2ecc71'
OVERLAY_ALPHA = 0.55


def _overlay(axis: plt.Axes, image: np.ndarray, plane: np.ndarray, title: str) -> None:
  axis.imshow(image)
  axis.imshow(plane, cmap='jet', alpha=OVERLAY_ALPHA, vmin=0.0, vmax=max(float(plane.max()), 1e-6))
  axis.set_title(title, fontsize=12, pad=8)
  axis.axis('off')


def render_detection_panels(
  image: np.ndarray,
  faces: list[FaceBox],
  center_heatmap: np.ndarray,
  size_map: np.ndarray,
  show_scores: bool = True,
) -> Figure:
  """Side-by-side panels: detections, center heatmap, size map (``√(w·h)``)."""
  height_px, width_px = image.shape[:2]
  panel_width = PANEL_HEIGHT_IN * width_px / height_px
  panels = 3
  fig_width = panels * panel_width + (panels - 1) * PANEL_GAP_IN
  fig_height = PANEL_HEIGHT_IN + TITLE_HEIGHT_IN
  fig = plt.figure(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)
  axes = [
    fig.add_axes(
      [
        index * (panel_width + PANEL_GAP_IN) / fig_width,
        0.0,
        panel_width / fig_width,
        PANEL_HEIGHT_IN / fig_height,
      ],
    )
    for index in range(panels)
  ]

  axes[0].imshow(image)
  for face in faces:
    axes[0].add_patch(
      Rectangle(
        (face.x1, face.y1),
        face.width,
        face.height,
        fill=False,
        edgecolor=BOX_COLOR,
        linewidth=1.5,
      ),
    )
    if show_scores:
      axes[0].text(
        face.x1,
        max(0.0, face.y1 - 3),
        f'{face.score:.2f}',
        color=BOX_COLOR,
        fontsize=8,
        va='bottom',
      )
  axes[0].set_title('Detections', fontsize=12, pad=8)
  axes[0].axis('off')

  _overlay(axes[1], image, center_heatmap, 'Center heatmap')
  _overlay(axes[2], image, size_map, 'Size map')
  return fig
