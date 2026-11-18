"""Stacked hourglass face detector: ``(B, 3, 512, 512)`` RGB → center, size, offset maps.

Center: one peak per face. Size: ``(width, height)`` in input pixels at each peak.
Offset: sub-cell centre position. All maps are ``H/4 × W/4``.

~0.64M parameters (64-channel residuals, 2 stacks). By default only the final
stack is returned; ``return_all_stacks=True`` exposes every stack for training.
Module attribute names are part of the checkpoint format.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_FILTERS = 64
DEFAULT_STACKS = 2
BN_MOMENTUM = 0.01
BN_EPS = 1e-3
# sigmoid(-2.19) ≈ 0.1: start the center head near the rare-positive prior.
CENTER_PRIOR_BIAS = -2.19

Heads = dict[str, torch.Tensor]


def batch_norm(channels: int) -> nn.BatchNorm2d:
  """BatchNorm2d with this architecture's momentum / epsilon."""
  return nn.BatchNorm2d(channels, eps=BN_EPS, momentum=BN_MOMENTUM)


class Conv2dSame(nn.Conv2d):
  """Conv2d that keeps the spatial size for stride 1 (``same`` padding).

  For stride > 1 the extra row / column (when needed) goes on the bottom-right;
  symmetric padding would shift every feature map by half an input pixel.
  """

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    padding: list[int] = []
    for size, kernel, stride in reversed(
      list(zip(x.shape[-2:], self.kernel_size, self.stride)),
    ):
      total = max((math.ceil(size / stride) - 1) * stride + kernel - size, 0)
      padding.extend([total // 2, total - total // 2])
    return super().forward(F.pad(x, padding))


class BasicResidual(nn.Module):
  """Conv3×3-BN-ReLU-Conv3×3-BN + identity, then ReLU."""

  def __init__(self, channels: int) -> None:
    super().__init__()
    self.conv1 = Conv2dSame(channels, channels, 3)
    self.bn1 = batch_norm(channels)
    self.conv2 = Conv2dSame(channels, channels, 3)
    self.bn2 = batch_norm(channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = F.relu(self.bn1(self.conv1(x)))
    return F.relu(self.bn2(self.conv2(out)) + x)


class HourglassModule(nn.Module):
  """Residual block, then ``depth`` × (max-pool, residual); nearest-upsample + skip."""

  def __init__(self, filters: int, depth: int) -> None:
    super().__init__()
    self.entry = BasicResidual(filters)
    self.down = nn.ModuleList([BasicResidual(filters) for _ in range(depth)])
    self.pool = nn.MaxPool2d(2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    current = self.entry(x)
    skips = [current]
    for level, block in enumerate(self.down):
      current = block(self.pool(current))
      if level < len(self.down) - 1:
        skips.append(current)
    for skip in reversed(skips):
      current = F.interpolate(current, scale_factor=2, mode='nearest') + skip
    return current


class FaceRadarNet(nn.Module):
  """Stacked hourglass with center / size / offset heads.

  Stem: 7×7/2 conv (``stem_filters``) → residual → 3×3/2 conv (``filters``) →
  residual. Each stack: hourglass → 3×3 conv-BN-ReLU features → three 1×1 heads.
  Between stacks the next input is
  ``current + conv1×1(features) + conv1×1(center logits)``.

  Outputs per stack:

  - ``center``: ``(B, 1, H/4, W/4)`` logits; sigmoid gives face-centre confidence
  - ``size``: ``(B, 2, H/4, W/4)`` face width / height in input pixels
  - ``offset``: ``(B, 2, H/4, W/4)`` centre residual inside the cell, in cells
  """

  def __init__(
    self,
    stacks: int = DEFAULT_STACKS,
    filters: int = DEFAULT_FILTERS,
    stem_filters: int = 32,
    depth: int = 2,
  ) -> None:
    super().__init__()
    self.stacks = int(stacks)
    self.filters = int(filters)
    self.stem = nn.Sequential(
      Conv2dSame(3, stem_filters, 7, stride=2),
      batch_norm(stem_filters),
      nn.ReLU(),
      BasicResidual(stem_filters),
      Conv2dSame(stem_filters, filters, 3, stride=2),
      batch_norm(filters),
      nn.ReLU(),
      BasicResidual(filters),
    )
    self.hourglasses = nn.ModuleList(
      [HourglassModule(filters, depth) for _ in range(stacks)],
    )
    self.features = nn.ModuleList(
      [
        nn.Sequential(
          Conv2dSame(filters, filters, 3),
          batch_norm(filters),
          nn.ReLU(),
        )
        for _ in range(stacks)
      ],
    )
    self.centerHeads = nn.ModuleList([Conv2dSame(filters, 1, 1) for _ in range(stacks)])
    self.sizeHeads = nn.ModuleList([Conv2dSame(filters, 2, 1) for _ in range(stacks)])
    self.offsetHeads = nn.ModuleList([Conv2dSame(filters, 2, 1) for _ in range(stacks)])
    self.mergeFeatures = nn.ModuleList(
      [Conv2dSame(filters, filters, 1) for _ in range(stacks - 1)],
    )
    self.mergeCenter = nn.ModuleList(
      [Conv2dSame(1, filters, 1) for _ in range(stacks - 1)],
    )
    self._init_weights()

  def _init_weights(self) -> None:
    """Glorot-uniform kernels, zero biases; center head biased toward rare peaks."""
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
    for head in self.centerHeads:
      nn.init.constant_(head.bias, CENTER_PRIOR_BIAS)

  def forward(
    self,
    x: torch.Tensor,
    return_all_stacks: bool = False,
  ) -> Heads | list[Heads]:
    """``(B, 3, H, W)`` RGB in ``[0, 1]`` → ``{'center', 'size', 'offset'}`` maps."""
    current = self.stem(x)
    outputs: list[Heads] = []
    for stack in range(self.stacks):
      features = self.features[stack](self.hourglasses[stack](current))
      center = self.centerHeads[stack](features)
      outputs.append(
        {
          'center': center,
          'size': self.sizeHeads[stack](features),
          'offset': self.offsetHeads[stack](features),
        },
      )
      if stack < self.stacks - 1:
        current = (
          current
          + self.mergeFeatures[stack](features)
          + self.mergeCenter[stack](center)
        )
    return outputs if return_all_stacks else outputs[-1]


def build_face_radar(
  stacks: int = DEFAULT_STACKS,
  filters: int | None = None,
) -> FaceRadarNet:
  """Construct the detector. ``filters=None`` uses ``DEFAULT_FILTERS``.

  Raises:
    ValueError: ``stacks < 1``.
  """
  if stacks < 1:
    raise ValueError(f'stacks must be >= 1, got {stacks}')
  width = DEFAULT_FILTERS if filters is None else int(filters)
  return FaceRadarNet(stacks=stacks, filters=width)
