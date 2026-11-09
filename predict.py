"""Command line: python3 predict.py photo.jpg [more.jpg ...] [--overlay out_dir]"""
import argparse
from pathlib import Path

import cv2

from anemia_eye import AnemiaEye

LABELS = {'green': 'low probability of anemia',
          'orange': 'uncertain: a blood test is advised',
          'red': 'high probability of anemia: a blood test is strongly advised'}

ap = argparse.ArgumentParser(description='Anemia screening from a photo of the lower eyelid.')
ap.add_argument('images', nargs='+', type=Path)
ap.add_argument('--overlay', type=Path, help='folder where to save the segmentation overlay')
args = ap.parse_args()

eye = AnemiaEye()
for path in args.images:
    r = eye.predict(cv2.imread(str(path)))
    if not r.ok:
        print(f'{path.name}: rejected ({r.error})')
        continue
    print(f'{path.name}: {r.triage.upper()} ({LABELS[r.triage]}), p={r.probability:.2f}, a_norm={r.a_norm:.2f}')
    if args.overlay:
        args.overlay.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.overlay / f'{path.stem}_overlay.jpg'), r.overlay)
