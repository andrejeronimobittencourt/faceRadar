"""Training loop: data loaders, epochs, checkpointing and resume."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from faceradar.data.augment import ZoomCropConfig
from faceradar.data.csv_format import FaceRecord
from faceradar.data.dataset import FaceBoxDataset, seed_worker
from faceradar.training.checkpoints import last_checkpoint_path, save_checkpoint
from faceradar.training.losses import LOSS_PARTS, LossWeights, stacked_detection_loss

# Pull the running loss to the host every N batches (avoids a CUDA sync per step).
LOG_EVERY = 50


def resolve_device(choice: str) -> torch.device:
  """``auto`` → CUDA when available, else CPU. ``cuda`` fails loudly if unavailable."""
  if choice == 'cpu':
    return torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Device: cuda ({torch.cuda.get_device_name(0)})')
    return device
  if choice == 'cuda':
    raise RuntimeError('--device cuda requested but torch.cuda.is_available() is False')
  print('Device: cpu (CUDA not available)')
  return torch.device('cpu')


def seed_everything(seed: int) -> None:
  """Seed torch (CPU + CUDA) for reproducible initialization and shuffling."""
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def build_loaders(
  train_records: list[FaceRecord],
  val_records: list[FaceRecord],
  train_images: Path,
  val_images: Path,
  batch_size: int,
  workers: int,
  augment: bool,
  flip_prob: float,
  zoom: ZoomCropConfig,
  zoom_extra: float,
  min_face_side: float,
  seed: int,
  device: torch.device,
) -> tuple[DataLoader, DataLoader]:
  """Augmented, shuffled train loader (full batches only) + deterministic val loader."""
  train_ds = FaceBoxDataset(
    train_records,
    train_images,
    min_face_side=min_face_side,
    augment=augment,
    flip_prob=flip_prob,
    zoom=zoom,
    zoom_extra=zoom_extra,
    seed=seed,
  )
  val_ds = FaceBoxDataset(val_records, val_images, min_face_side=min_face_side, seed=seed + 1)
  generator = torch.Generator()
  generator.manual_seed(seed)
  common = {
    'batch_size': batch_size,
    'num_workers': workers,
    'pin_memory': device.type == 'cuda',
    'persistent_workers': workers > 0,
    'worker_init_fn': seed_worker,
  }
  return (
    DataLoader(train_ds, shuffle=True, drop_last=True, generator=generator, **common),
    DataLoader(val_ds, shuffle=False, **common),
  )


def build_scheduler(
  optimizer: torch.optim.Optimizer,
  epochs: int,
  min_lr: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
  """Per-epoch cosine anneal from the optimizer LR to ``min_lr``."""
  return torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, epochs),
    eta_min=min_lr,
  )


@dataclass(frozen=True)
class EpochLoss:
  """Sample-weighted mean final-stack losses of one pass: weighted total + unweighted parts."""

  total: float
  center: float
  size: float
  offset: float

  def describe(self) -> str:
    return (
      f'{self.total:.5f} (ctr={self.center:.4f} size={self.size:.4f} off={self.offset:.4f})'
    )


def run_epoch(
  model: nn.Module,
  loader: DataLoader,
  weights: LossWeights,
  device: torch.device,
  use_amp: bool,
  desc: str,
  optimizer: torch.optim.Optimizer | None = None,
  scaler: torch.amp.GradScaler | None = None,
  all_stacks: bool = True,
) -> EpochLoss:
  """One pass over ``loader``; trains when ``optimizer`` is given.

  ``all_stacks`` selects intermediate supervision for the training objective;
  the returned losses are the final stack's either way.
  """
  training = optimizer is not None
  model.train(training)
  names = ('total', *LOSS_PARTS)
  sums = {name: torch.zeros((), device=device) for name in names}
  count = 0
  window = torch.zeros((), device=device)
  window_steps = 0
  bar = tqdm(loader, desc=desc, leave=False, unit='batch', dynamic_ncols=True)
  with torch.set_grad_enabled(training):
    for step, batch in enumerate(bar, start=1):
      batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
      with torch.autocast(device.type, enabled=use_amp):
        outputs = model(batch['image'], return_all_stacks=True)
      objective, loss, parts = stacked_detection_loss(
        outputs,
        batch,
        weights,
        all_stacks=training and all_stacks,
      )
      if training:
        if not torch.isfinite(objective):
          raise RuntimeError(f'Non-finite training loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(objective).backward()
        scaler.step(optimizer)
        scaler.update()
      batch_size = batch['image'].shape[0]
      for name, value in {'total': loss, **parts}.items():
        sums[name] = sums[name] + value * batch_size
      count += batch_size
      window = window + loss
      window_steps += 1
      if step % LOG_EVERY == 0 or step == len(loader):
        bar.set_postfix(loss=f'{(window / max(1, window_steps)).item():.5f}')
        window = torch.zeros((), device=device)
        window_steps = 0
  means = {name: float((value / max(1, count)).item()) for name, value in sums.items()}
  return EpochLoss(**means)


def fit(
  *,
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  weights: LossWeights,
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
  scaler: torch.amp.GradScaler,
  device: torch.device,
  use_amp: bool,
  epochs: int,
  start_epoch: int,
  best_val_loss: float,
  output_path: Path,
  meta: dict[str, object],
  all_stacks: bool = True,
) -> float:
  """Train for ``start_epoch..epochs``; return the best validation loss.

  The best checkpoint is written when the final stack's validation total
  improves; the ``.last`` state every epoch.
  """
  output_path = Path(output_path)
  last_path = last_checkpoint_path(output_path)
  for epoch in range(start_epoch, epochs + 1):
    started = time.perf_counter()
    train_loss = run_epoch(
      model,
      train_loader,
      weights,
      device,
      use_amp,
      f'train {epoch}/{epochs}',
      optimizer=optimizer,
      scaler=scaler,
      all_stacks=all_stacks,
    )
    val_loss = run_epoch(model, val_loader, weights, device, use_amp, f'val {epoch}/{epochs}')
    scheduler.step()
    tqdm.write(
      f'epoch {epoch:03d}/{epochs}  train={train_loss.total:.5f}  '
      f'val={val_loss.describe()}  lr={optimizer.param_groups[0]["lr"]:.2e}  '
      f'{time.perf_counter() - started:.1f}s'
    )
    if val_loss.total < best_val_loss:
      best_val_loss = val_loss.total
      save_checkpoint(
        output_path,
        {**meta, 'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss.total},
      )
      tqdm.write(f'  saved best → {output_path}')
    save_checkpoint(
      last_path,
      {
        **meta,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
      },
    )
  return best_val_loss


def restore_state(
  state: dict,
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
  scaler: torch.amp.GradScaler,
  epochs: int,
) -> tuple[int, float]:
  """Load a ``.last`` state; return ``(next_epoch, best_val_loss)``.

  Raises:
    RuntimeError: The cosine schedule was planned for a different number of
      epochs. The restored schedule would ignore the new ``--epochs`` (and stays
      at the minimum LR once finished), so changing it needs ``--fresh``.
  """
  model.load_state_dict(state['model'])
  optimizer.load_state_dict(state['optimizer'])
  scheduler.load_state_dict(state['scheduler'])
  if 'scaler' in state:
    scaler.load_state_dict(state['scaler'])
  planned = int(scheduler.T_max)
  if planned != max(1, epochs):
    raise RuntimeError(
      f'The checkpoint\'s cosine LR schedule was planned for {planned} epochs, '
      f'but --epochs is {epochs}. Resume with --epochs {planned}, or pass --fresh '
      'to restart with a new schedule.'
    )
  return int(state['epoch']) + 1, float(state.get('best_val_loss', float('inf')))
