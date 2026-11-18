#!/usr/bin/env python3
"""Detect every face in one image: boxes, center heatmap and size map."""

import argparse
from pathlib import Path

from faceradar.config import DecodeParams, Paths
from faceradar.inference.pipeline import run_infer_cli


def parse_args() -> argparse.Namespace:
  paths = Paths()
  defaults = DecodeParams()
  parser = argparse.ArgumentParser(description='Detect faces in one image.')
  parser.add_argument('--image', required=True, help='Path to the input image.')
  parser.add_argument(
    '--model',
    default=str(paths.face_radar),
    help=f'Path to a trained .pt checkpoint (default: {paths.face_radar}).',
  )
  parser.add_argument('--output', help='Optional path to save the plot.')
  parser.add_argument(
    '--conf',
    type=float,
    default=defaults.conf_threshold,
    help='Minimum center confidence for a face.',
  )
  parser.add_argument(
    '--nms-iou',
    type=float,
    default=defaults.nms_iou,
    help='Boxes overlapping a stronger one by at least this IoU are dropped.',
  )
  parser.add_argument(
    '--no-scores',
    action='store_true',
    help='Draw boxes without confidence labels.',
  )
  parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  image_path = Path(args.image)
  model_path = Path(args.model)

  if not model_path.is_file():
    raise FileNotFoundError(f'Model file not found: {model_path}')
  if not image_path.is_file():
    raise FileNotFoundError(f'Image file not found: {image_path}')

  run_infer_cli(
    image_path,
    model_path,
    args.output,
    args.device,
    DecodeParams(conf_threshold=args.conf, nms_iou=args.nms_iou),
    show_scores=not args.no_scores,
  )


if __name__ == '__main__':
  main()
