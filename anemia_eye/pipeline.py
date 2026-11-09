"""Anemia screening from a photo of the lower eyelid (palpebral conjunctiva).

Pipeline, identical to the one running in the Yako app:
  1. quality check (blur, exposure, resolution)
  2. conjunctiva segmentation with a U-Net (ResNet-50 encoder)
  3. geometric sanity checks on the mask
  4. skin-anchored colour measure: grey-world white balance on the periocular
     skin, then a_norm = mean a* (CIELAB) of the conjunctiva - mean a* of the skin
  5. logistic regression on a_norm -> probability -> green / orange / red

A pale conjunctiva (low a_norm) means a higher probability of anemia.
Research prototype for triage. It is not a diagnostic device.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIBRATION = json.loads((HERE / 'model_calibration.json').read_text())

WEIGHTS_URL = os.environ.get(
    'ANEMIA_EYE_WEIGHTS_URL',
    'https://huggingface.co/pk180/anemiaeye-models/resolve/main/unet_resnet50_v2_fullframe.pt')
WEIGHTS_SHA256 = '74778a8ddb9ccf7a83cdfd1d653b69339a51828c50420571054323b20d0b4fc0'
WEIGHTS_PATH = Path(os.environ.get('ANEMIA_EYE_WEIGHTS', HERE.parent / 'weights' / 'unet_resnet50_v2_fullframe.pt'))

EPS = 1e-6
CANVAS = 512          # every measure is taken on a 512x512 canvas (calibration depends on it)
UNET_INPUT = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Result:
    ok: bool
    error: Optional[str] = None
    probability: Optional[float] = None   # probability of anemia (Hb below WHO threshold)
    a_norm: Optional[float] = None
    triage: Optional[str] = None          # green | orange | red
    overlay: Optional[np.ndarray] = None  # BGR image: conjunctiva in green, skin anchor in blue
    debug: dict = field(default_factory=dict)


def download_weights(path: Path = WEIGHTS_PATH) -> Path:
    """Downloads the U-Net weights (125 MB) once and checks their SHA-256."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part')
    print(f'Downloading segmentation weights from {WEIGHTS_URL} ...')
    # requests + certifi: urllib fails on some Python installs (macOS) without system certificates.
    import requests
    with requests.get(WEIGHTS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        tmp.unlink()
        raise RuntimeError(f'Unexpected weights checksum {digest}')
    tmp.rename(path)
    return path


class AnemiaEye:
    def __init__(self, weights: Path = WEIGHTS_PATH, device: Optional[str] = None):
        import torch
        import segmentation_models_pytorch as smp
        self.torch = torch
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.unet = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                             in_channels=3, classes=1, activation=None)
        state = torch.load(download_weights(weights), map_location='cpu', weights_only=True)
        self.unet.load_state_dict(state)
        self.unet.to(self.device).eval()

    # ---------------------------------------------------------------- steps
    @staticmethod
    def quality_check(img: np.ndarray) -> Optional[str]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        sharpness = cv2.Laplacian(gray[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.CV_64F).var()
        lum = gray.mean()
        if sharpness < 1.0:
            return 'Image too blurry: hold the phone still'
        if lum < 20:
            return 'Image too dark: move closer to a light source'
        if lum > 210 and (gray > 245).mean() > 0.30:
            return 'Image overexposed: avoid direct flash'
        if min(h, w) < 400:
            return f'Resolution too low ({w}x{h})'
        return None

    def segment(self, img512: np.ndarray):
        """Largest connected component of the U-Net mask, and its mean confidence."""
        torch = self.torch
        rgb = cv2.resize(cv2.cvtColor(img512, cv2.COLOR_BGR2RGB), (UNET_INPUT, UNET_INPUT)).astype(np.float32) / 255.
        x = torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            logit = self.unet(x.to(self.device))[0, 0].cpu().numpy()
        prob = 1 / (1 + np.exp(-cv2.resize(logit, (CANVAS, CANVAS), interpolation=cv2.INTER_LINEAR)))
        raw = prob > 0.5
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8) * 255, connectivity=8)
        if n < 2:
            return raw, float(prob[raw].mean()) if raw.any() else 0.0
        mask = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return mask, float(prob[mask].mean())

    @staticmethod
    def measure(img512: np.ndarray, mask: np.ndarray):
        """Skin-anchored a_norm. Returns (a_norm, conjunctiva pixels, skin pixels, debug) or None."""
        lum = img512.mean(2)
        dil = lambda k: cv2.dilate(mask.astype(np.uint8) * 255,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        # Skin anchor: a ring 10-40 px away from the mask. A ring glued to the mask falls on the
        # eyelid rim, which also gets paler with anemia and cancels the signal.
        ring = dil(80) & ~dil(20) & (lum > 15) & (lum < 195)
        conj = mask & (lum > 12) & (lum < 240)

        # Specular reflections show the colour of the light source, not of the tissue:
        # they pull a* towards 0, i.e. towards "anemic". Remove them.
        hsv = cv2.cvtColor(img512, cv2.COLOR_BGR2HSV)
        spec = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 60)
        if (conj & ~spec).sum() >= 30:
            conj = conj & ~spec

        # Keep only ring pixels whose colour matches certain skin (a thin band next to the mask):
        # on wide shots the ring can reach a finger, hair or the wall.
        # Diagonal Mahalanobis distance in Lab with a robust (MAD) spread, threshold 2.5.
        near = dil(16) & ~mask & (lum > 15) & (lum < 195)
        skin, rejected = ring, 0.0
        if near.sum() >= 40 and ring.sum() >= 40:
            lab = cv2.cvtColor(img512, cv2.COLOR_BGR2Lab).astype(float)
            d = np.zeros(mask.shape, float)
            for i in range(3):
                c = lab[:, :, i]
                mu = np.median(c[near])
                sd = max(1.4826 * np.median(np.abs(c[near] - mu)), 3.0)
                d += ((c - mu) / sd) ** 2
            kept = ring & (np.sqrt(d) <= 2.5)
            if kept.sum() >= 40:
                skin, rejected = kept, float(1 - kept.sum() / ring.sum())
        if conj.sum() < 30 or skin.sum() < 40:
            return None

        # Grey-world white balance anchored on the skin, then a* difference.
        means = [img512[:, :, ch][skin].mean() for ch in range(3)]
        grey = sum(means) / 3
        wb = img512.astype(float)
        for ch in range(3):
            wb[:, :, ch] = np.clip(wb[:, :, ch] * grey / (means[ch] + EPS), 0, 255)
        a = cv2.cvtColor(wb.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(float)[:, :, 1] - 128
        a_conj, a_skin = float(a[conj].mean()), float(a[skin].mean())
        debug = {'a_conj': round(a_conj, 2), 'a_skin': round(a_skin, 2), 'n_conj_px': int(conj.sum()),
                 'n_skin_px': int(skin.sum()), 'skin_ring_rejected': round(rejected, 2),
                 'specular_px': int((mask & spec).sum())}
        return a_conj - a_skin, conj, skin, debug

    @staticmethod
    def score(a_norm: float):
        c = CALIBRATION
        z = (a_norm - c['mean']) / c['scale']
        p = float(1 / (1 + np.exp(-(c['coef'] * z + c['intercept']))))
        return p, 'green' if p < c['t_green'] else 'orange' if p < c['t_red'] else 'red'

    # ---------------------------------------------------------------- entry point
    def predict(self, img: np.ndarray) -> Result:
        if img is None:
            return Result(False, 'Unreadable image')
        err = self.quality_check(img)
        if err:
            return Result(False, err)
        img512 = cv2.resize(img, (CANVAS, CANVAS))
        mask, conf = self.segment(img512)
        n = int(mask.sum())
        if n < 50:
            return Result(False, 'Conjunctiva not found: pull the lower eyelid down and centre the eye')
        ys, xs = np.where(mask)
        cy = ys.mean() / CANVAS
        aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1 + EPS)
        if cy < 0.25 or cy > 0.80:
            return Result(False, 'Detected area is off-centre: centre the eye in the frame')
        if n > CANVAS * CANVAS * 0.18 or aspect < 1.1:
            return Result(False, 'Conjunctiva not found: the pink area should be a horizontal band')
        m = self.measure(img512, mask)
        if m is None:
            return Result(False, 'Not enough skin around the eye in the frame')
        a_norm, conj, skin, debug = m
        p, triage = self.score(a_norm)

        h, w = img.shape[:2]
        up = lambda x: cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        overlay = img.copy()
        cf, sf = up(conj), up(skin)
        overlay[cf] = (overlay[cf] * 0.4 + np.array([0, 200, 80]) * 0.6).astype(np.uint8)
        overlay[sf] = (overlay[sf] * 0.5 + np.array([200, 80, 0]) * 0.5).astype(np.uint8)
        debug['unet_confidence'] = round(conf, 3)
        return Result(True, None, round(p, 3), round(a_norm, 3), triage, overlay, debug)

    def predict_series(self, images) -> Result:
        """Several photos of the same eye taken a moment apart (the app takes 3, 0.25 s apart).

        Keeps the photo with the median a_norm, so its image, probability and colour stay consistent;
        with two valid photos, the more cautious one (higher probability). debug['stable'] says whether
        all valid photos give the same colour. In the cohort, two single photos agree on the colour in
        83% of cases, two medians of 3 in 91%.
        """
        results = [self.predict(img) for img in images]
        ok = sorted((r for r in results if r.ok), key=lambda r: r.a_norm)
        if not ok:
            return next((r for r in results if r.error), Result(False, 'No usable photo'))
        pick = max(ok, key=lambda r: r.probability) if len(ok) == 2 else ok[(len(ok) - 1) // 2]
        pick.debug.update(n_photos=len(ok), stable=len({r.triage for r in ok}) == 1)
        return pick
