"""Penalty-reduced focal loss on the center map + masked L1 on size / offset."""

from __future__ import annotations

from dataclasses import dataclass

import torch

LOSS_PARTS: tuple[str, ...] = ('center', 'size', 'offset')


@dataclass(frozen=True)
class LossWeights:
  """Weights of the three loss terms in the total."""

  center: float = 1.0
  size: float = 0.1
  offset: float = 1.0


def focal_center_loss(
  logits: torch.Tensor,
  target: torch.Tensor,
  alpha: float = 2.0,
  beta: float = 4.0,
) -> torch.Tensor:
  """CornerNet / CenterNet focal loss, normalized by the number of peak cells.

  Cells where ``target == 1`` are positives; every other cell is a negative whose
  penalty is reduced by ``(1 - target) ** beta`` near a peak.
  """
  prediction = torch.sigmoid(logits.float()).clamp(1e-4, 1.0 - 1e-4)
  positive = target.eq(1).float()
  negative = target.lt(1).float()
  positive_loss = -((1.0 - prediction) ** alpha) * torch.log(prediction) * positive
  negative_loss = (
    -(prediction**alpha) * torch.log(1.0 - prediction) * (1.0 - target) ** beta * negative
  )
  return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp(min=1.0)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Mean absolute error over the centre cells only (``mask`` is ``(B, h, w)``)."""
  weights = mask[:, None].expand_as(prediction)
  return (torch.abs(prediction.float() - target) * weights).sum() / weights.sum().clamp(min=1.0)


def detection_loss(
  output: dict[str, torch.Tensor],
  batch: dict[str, torch.Tensor],
  weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Weighted total + the detached unweighted parts (``center``, ``size``, ``offset``)."""
  parts = {
    'center': focal_center_loss(output['center'], batch['center']),
    'size': masked_l1(output['size'], batch['size'], batch['mask']),
    'offset': masked_l1(output['offset'], batch['offset'], batch['mask']),
  }
  total = (
    weights.center * parts['center']
    + weights.size * parts['size']
    + weights.offset * parts['offset']
  )
  return total, {name: value.detach() for name, value in parts.items()}


def stacked_detection_loss(
  outputs: list[dict[str, torch.Tensor]],
  batch: dict[str, torch.Tensor],
  weights: LossWeights,
  all_stacks: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
  """Training objective + the final stack's detached total and parts.

  With ``all_stacks`` every stack is supervised (losses summed). The returned
  total / parts are always from the final stack (what inference uses).
  """
  final_total, final_parts = detection_loss(outputs[-1], batch, weights)
  objective = final_total
  if all_stacks:
    for output in outputs[:-1]:
      objective = objective + detection_loss(output, batch, weights)[0]
  return objective, final_total.detach(), final_parts
