#!/usr/bin/env python3
"""Train the FaceRadar detector on a face-box CSV dataset (resumes automatically)."""

import argparse
from pathlib import Path

import torch

from faceradar.config import Paths, TrainingConfig
from faceradar.data.augment import ZoomCropConfig
from faceradar.data.csv_format import load_face_csv
from faceradar.models.hourglass import build_face_radar
from faceradar.training.checkpoints import checkpoint_meta, read_resume_state
from faceradar.training.losses import LossWeights
from faceradar.training.loop import (
  build_loaders,
  build_scheduler,
  fit,
  resolve_device,
  restore_state,
  seed_everything,
)


def parse_args() -> argparse.Namespace:
  paths = Paths()
  cfg = TrainingConfig()
  parser = argparse.ArgumentParser(
    description='Train the multi-face center / size / offset hourglass.',
  )
  parser.add_argument('--train-csv', default=str(paths.train_csv), help='Training face boxes.')
  parser.add_argument('--train-images', default=str(paths.train_images))
  parser.add_argument('--val-csv', default=str(paths.val_csv), help='Validation face boxes.')
  parser.add_argument('--val-images', default=str(paths.val_images))
  parser.add_argument(
    '--output',
    default=str(paths.face_radar),
    help='Best checkpoint path; the resumable state goes to <name>.last.pt.',
  )
  parser.add_argument('--epochs', type=int, default=cfg.epochs)
  parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
  parser.add_argument('--lr', type=float, default=cfg.learning_rate)
  parser.add_argument('--min-lr', type=float, default=cfg.min_learning_rate)
  parser.add_argument('--stacks', type=int, default=cfg.stacks)
  parser.add_argument('--filters', type=int, default=cfg.filters)
  parser.add_argument('--center-weight', type=float, default=cfg.center_weight)
  parser.add_argument('--size-weight', type=float, default=cfg.size_weight)
  parser.add_argument('--offset-weight', type=float, default=cfg.offset_weight)
  parser.add_argument(
    '--min-face-side',
    type=float,
    default=cfg.min_face_side,
    help='Faces smaller than this on the 512 input (px) get no target.',
  )
  parser.add_argument('--flip-prob', type=float, default=cfg.flip_prob)
  parser.add_argument(
    '--zoom-extra',
    type=float,
    default=cfg.zoom_extra,
    help='Extra close-up crops as a fraction of the train set (0 disables).',
  )
  parser.add_argument(
    '--zoom-face-share',
    type=float,
    default=cfg.zoom_face_share,
    help='Share of extra crops centred on a face; the rest are random patches.',
  )
  parser.add_argument(
    '--final-stack-only',
    action='store_true',
    help='Supervise only the last hourglass stack.',
  )
  parser.add_argument('--workers', type=int, default=cfg.workers)
  parser.add_argument('--seed', type=int, default=cfg.seed)
  parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
  parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision on CUDA.')
  parser.add_argument('--no-aug', action='store_true', help='Disable training augmentation.')
  parser.add_argument('--fresh', action='store_true', help='Ignore existing checkpoints.')
  args = parser.parse_args()
  if args.epochs < 1 or args.batch_size < 1:
    parser.error('--epochs and --batch-size must be >= 1')
  if not 0.0 <= args.flip_prob <= 1.0:
    parser.error('--flip-prob must be in [0, 1]')
  if not 0.0 <= args.zoom_face_share <= 1.0:
    parser.error('--zoom-face-share must be in [0, 1]')
  if args.zoom_extra < 0.0:
    parser.error('--zoom-extra must be >= 0')
  return args


def main() -> None:
  args = parse_args()
  device = resolve_device(args.device)
  seed_everything(args.seed)
  output_path = Path(args.output)
  zoom = ZoomCropConfig(face_share=args.zoom_face_share)
  zoom_extra = 0.0 if args.no_aug else args.zoom_extra
  all_stacks = not args.final_stack_only

  train_records = load_face_csv(args.train_csv)
  val_records = load_face_csv(args.val_csv)
  print(
    f'Train images: {len(train_records)} ({sum(len(r.boxes) for r in train_records)} faces)  '
    f'Val images: {len(val_records)} ({sum(len(r.boxes) for r in val_records)} faces)'
  )
  train_loader, val_loader = build_loaders(
    train_records,
    val_records,
    Path(args.train_images),
    Path(args.val_images),
    batch_size=args.batch_size,
    workers=args.workers,
    augment=not args.no_aug,
    flip_prob=args.flip_prob,
    zoom=zoom,
    zoom_extra=zoom_extra,
    min_face_side=args.min_face_side,
    seed=args.seed,
    device=device,
  )

  model = build_face_radar(stacks=args.stacks, filters=args.filters).to(device)
  params = sum(p.numel() for p in model.parameters())
  print(
    f'Model params={params:,} stacks={args.stacks} filters={args.filters}  '
    f'supervision={"all stacks" if all_stacks else "final stack"}  '
    f'zoom_extra={zoom_extra:.2f} (face share {zoom.face_share:.2f})'
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  scheduler = build_scheduler(optimizer, args.epochs, args.min_lr)
  use_amp = device.type == 'cuda' and not args.no_amp
  scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
  meta = checkpoint_meta(
    stacks=args.stacks,
    filters=args.filters,
    intermediate_supervision=all_stacks,
    zoom=zoom,
    zoom_extra=zoom_extra,
  )

  start_epoch, best_val_loss = 1, float('inf')
  state = read_resume_state(output_path, meta, device, args.fresh)
  if state is not None:
    start_epoch, best_val_loss = restore_state(
      state,
      model,
      optimizer,
      scheduler,
      scaler,
      args.epochs,
    )
    if start_epoch > args.epochs:
      print(f'{output_path} already finished {args.epochs} epochs. Pass --fresh to retrain.')
      return
    print(f'Resuming at epoch {start_epoch} (best val={best_val_loss:.5f})')

  fit(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    weights=LossWeights(
      center=args.center_weight,
      size=args.size_weight,
      offset=args.offset_weight,
    ),
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
    use_amp=use_amp,
    epochs=args.epochs,
    start_epoch=start_epoch,
    best_val_loss=best_val_loss,
    output_path=output_path,
    meta=meta,
    all_stacks=all_stacks,
  )
  print(f'Done → {output_path}')


if __name__ == '__main__':
  main()
