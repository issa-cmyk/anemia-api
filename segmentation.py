"""
segmentation.py — Segmentation de la conjonctive palpébrale.

Cascade de méthodes (la première qui réussit est utilisée) :
  0. YOLOv8-seg dédié conjonctive (si USE_YOLO=True et modèle disponible)
  1. SAM2 (Segment Anything v2) — fallback généraliste
  2. U-Net léger pré-entraîné (unet_conjonctiva.pt)
  3. Heuristique couleur a* Lab (dernier recours)

Flag de configuration :
  USE_YOLO = True   → YOLO en première position (après entraînement)
  USE_YOLO = False  → pipeline SAM2 original (pendant phase de test)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ── Flag de configuration ─────────────────────────────────────────────────────
USE_YOLO = False

# ─────────────────────────────────────────────────────────────────────────────
# 0. U-Net ResNet50 — best_unet.pt (CIV, test Dice=0.805, IoU=0.681)
#    Entraîné sur 211 frames d'yeux CIV avec CLAHE + inpainting reflets
# ─────────────────────────────────────────────────────────────────────────────

_BEST_UNET_PATH   = Path(__file__).parent / "best_unet.pt"
_BEST_UNET_MODEL  = None
_BEST_UNET_LOADED = False

_UNET_IMG_SIZE = 256
_UNET_MEAN = [0.485, 0.456, 0.406]
_UNET_STD  = [0.229, 0.224, 0.225]


def _try_load_best_unet() -> bool:
    global _BEST_UNET_MODEL, _BEST_UNET_LOADED
    if _BEST_UNET_LOADED:
        return _BEST_UNET_MODEL is not None
    _BEST_UNET_LOADED = True
    if not _BEST_UNET_PATH.exists():
        return False
    try:
        import torch
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name    = "resnet50",
            encoder_weights = None,
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )
        sd = torch.load(_BEST_UNET_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        _BEST_UNET_MODEL = model
        return True
    except Exception:
        return False


def _preprocess_for_best_unet(img_bgr: np.ndarray):
    """Inpainting reflets + CLAHE + normalisation ImageNet → tensor (1,3,H,W)."""
    import torch
    # Inpainting reflets spéculaires
    spec_mask = np.all(img_bgr > 240, axis=2).astype(np.uint8)
    if spec_mask.sum() > 0:
        img_bgr = cv2.inpaint(img_bgr, spec_mask, 3, cv2.INPAINT_TELEA)
    # CLAHE par canal BGR
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = np.empty_like(img_bgr)
    for c in range(3):
        out[:, :, c] = clahe.apply(img_bgr[:, :, c])
    # BGR → RGB → resize → normalize
    img_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    img_r   = cv2.resize(img_rgb, (_UNET_IMG_SIZE, _UNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_f   = img_r.astype(np.float32) / 255.0
    mean    = np.array(_UNET_MEAN, dtype=np.float32)
    std     = np.array(_UNET_STD,  dtype=np.float32)
    tensor  = torch.from_numpy(((img_f - mean) / std).transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def _segment_with_best_unet(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Segmentation U-Net ResNet50 CIV (test Dice=0.805)."""
    if not _try_load_best_unet() or _BEST_UNET_MODEL is None:
        return None
    try:
        import torch
        h, w   = img_bgr.shape[:2]
        tensor = _preprocess_for_best_unet(img_bgr)
        with torch.no_grad():
            prob = torch.sigmoid(_BEST_UNET_MODEL(tensor)).squeeze().numpy()
        pred_256 = (prob > 0.5).astype(np.uint8) * 255
        mask     = cv2.resize(pred_256, (w, h), interpolation=cv2.INTER_NEAREST)
        # Garder uniquement le plus grand composant connexe
        n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
        if n_lbl > 1:
            best_lbl = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
            mask = (lbl_map == best_lbl).astype(np.uint8) * 255
        n_px = int((mask > 0).sum())
        fill = float(n_px / (h * w) * 100)
        if n_px < 300 or fill < 0.3:
            return None
        roi_bgr = img_bgr.copy()
        roi_bgr[mask == 0] = 0
        info: dict = {
            "method":       "unet_resnet50_civ",
            "roi_fill_pct": round(fill, 2),
            "n_pixels":     n_px,
        }
        if n_px > 10:
            px_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)[mask > 0]
            info["lab_a_mean"] = round(float(px_lab[:, 1].mean()), 1)
        return roi_bgr, mask, info
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 1. SAM2 via ultralytics
# ─────────────────────────────────────────────────────────────────────────────

_SAM_MODEL = None
_SAM_LOADED = False


def _try_load_sam():
    global _SAM_MODEL, _SAM_LOADED
    if _SAM_LOADED:
        return _SAM_MODEL is not None
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from ultralytics import SAM
        _SAM_MODEL = SAM("sam2_t.pt")   # tiny ~38 MB, téléchargé auto
        _SAM_LOADED = True
        return True
    except Exception:
        _SAM_LOADED = True
        return False


def check_conjunctiva_visibility(img_bgr: np.ndarray) -> tuple[bool, list[str]]:
    """
    Contrôle qualité rapide AVANT segmentation.

    Retourne (is_valid, issues).
    - is_valid=False → photo BLOQUÉE (conjonctive absente)
    - is_valid=True + issues non vide → avertissements (flou, luminosité) mais on continue
    """
    h, w = img_bgr.shape[:2]
    issues: list[str] = []

    # Avertissements non-bloquants : flou et luminosité
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    if lap < 20:
        issues.append("Image floue — maintenez le téléphone stable")
    if brightness < 30:
        issues.append("Image trop sombre — éclairez davantage")
    elif brightness > 245:
        issues.append("Image surexposée — évitez la lumière directe")

    # BLOQUANT : conjonctive non visible
    # La conjonctive exposée crée un signal a* > 129, S > 12
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    a = lab[:, :, 1]
    L = lab[:, :, 0]
    S = hsv[:, :, 1]

    candidate = (a > 129) & (L > 45) & (L < 225) & (S > 12)
    conj_pct = float(candidate.sum()) / (h * w) * 100

    if conj_pct < 0.10:
        issues.insert(0, "Conjonctive non visible — tirez la paupière inférieure vers le bas")
        return False, issues   # bloqué

    return True, issues   # valide (avec ou sans avertissements)


def _locate_eye_center(img_bgr: np.ndarray) -> tuple[int, int] | None:
    """
    Estime le centre de l'œil via la sclérotique (blanc, peu saturé).
    Contraintes strictes pour éviter de détecter l'ongle ou le fond.
    Retourne (cx, cy) en pixels, ou None si introuvable.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Seuils abaissés pour peau sombre : sclère peut apparaître < 150 L*
    sclera = (L > 120) & (S < 80) & (V > 110)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sclera_u8 = sclera.astype(np.uint8) * 255
    sclera_u8 = cv2.morphologyEx(sclera_u8, cv2.MORPH_CLOSE, k, iterations=2)

    n_lbl, lbl_map, stats, centroids = cv2.connectedComponentsWithStats(sclera_u8, 8)
    if n_lbl <= 1:
        return None

    # Sélectionner la composante la plus large dans la zone centrale de l'image
    # (exclut fond, front, doigt, coins extrêmes)
    best_lbl, best_score = -1, -1
    for lbl in range(1, n_lbl):
        area  = int(stats[lbl, cv2.CC_STAT_AREA])
        cy_l  = float(centroids[lbl][1])
        cx_l  = float(centroids[lbl][0])
        if area < 200:
            continue
        # Zone valide : 3%–85% verticalement, 3%–97% horizontalement
        # Contrainte large : gros plan d'œil peut avoir l'iris à < 15% du haut
        if cy_l < h * 0.03 or cy_l > h * 0.85:
            continue
        if cx_l < w * 0.03 or cx_l > w * 0.97:
            continue
        if area > best_score:
            best_score, best_lbl = area, lbl

    if best_lbl < 0:
        return None

    cx = int(centroids[best_lbl][0])
    cy = int(centroids[best_lbl][1])
    return (cx, cy)


def _find_conjunctiva_seed(img_bgr: np.ndarray) -> tuple[int, int]:
    """
    Trouve le meilleur pixel-seed pour la conjonctive palpébrale.

    Stratégie ancrée sur la position anatomique de l'œil :
    1. Localiser le centre de la sclérotique (blanc de l'œil)
    2. Chercher la zone la plus rose/rouge SOUS le centre de l'œil
       (la conjonctive palpébrale est toujours sous l'iris)
    3. Fallback : moitié basse de l'image si la sclérotique n'est pas visible
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    eye_center = _locate_eye_center(img_bgr)

    if eye_center is not None:
        ex, ey = eye_center
        # Zone horizontale : ±32% autour du centre oculaire, mais pas plus de 85%
        # à droite (pour exclure les arrière-plans colorés à droite du cadre)
        half_w = int(w * 0.32)
        sx0 = max(0, ex - half_w)
        sx1 = min(int(w * 0.85), ex + half_w)
        sy0 = max(0, ey)
        # Cap vertical adaptatif selon la position de l'œil
        if ey < h * 0.50:
            # Œil haut (gros plan) : cap 65% pour éviter le doigt en bas
            sy1 = min(int(h * 0.65), ey + int(h * 0.25))
        else:
            # Œil bas (portrait plein visage) : 22% sous l'iris
            sy1 = min(int(h * 0.88), ey + int(h * 0.22))
    else:
        # Fallback : sclérotique non trouvée → chercher dans la zone centrale
        # (gros plan d'œil : conjonctive retournée occupe le centre ou bas de l'image)
        sx0, sx1 = int(w * 0.03), int(w * 0.97)
        sy0, sy1 = int(h * 0.15), int(h * 0.80)

    zone = img_bgr[sy0:sy1, sx0:sx1]
    if zone.size == 0:
        return (w // 2, int(h * 0.75))

    z_lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB).astype(np.float32)
    z_hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV).astype(np.float32)
    a = z_lab[:, :, 1]
    L = z_lab[:, :, 0]
    S = z_hsv[:, :, 1]

    score = (a - 128) * (S / 255.0)
    score[L < 20]  = 0   # exclut uniquement le noir pur (iris, fond)
    score[L > 210] = 0   # exclut surexposition / ongle très blanc
    score[S < 8]   = 0   # exclut pixels quasi-neutres (peau grise)
    score[S > 200] = 0   # exclut arrière-plans très saturés (vêtements)

    if score.max() <= 0:
        # Dernier recours : pixel le plus rougeâtre sans contrainte saturation
        score2 = (a - 128).clip(0)
        score2[L < 15] = 0
        score2[L > 225] = 0
        if score2.max() > 0:
            iy, ix = np.unravel_index(score2.argmax(), score2.shape)
            return (sx0 + ix, sy0 + iy)
        return (sx0 + zone.shape[1] // 2, sy0 + zone.shape[0] // 2)

    iy, ix = np.unravel_index(score.argmax(), score.shape)
    return (sx0 + ix, sy0 + iy)


def _segment_with_sam(
    img_bgr: np.ndarray,
    manual_seed: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, dict]:
    """
    SAM2 avec 1 seul point POSITIF — aucun point négatif.
    Si manual_seed est fourni, il est utilisé à la place de l'auto-détection.
    Le post-traitement adaptatif filtre sclère et non-conjonctive.
    """
    if not _try_load_sam() or _SAM_MODEL is None:
        return None, {}

    h, w = img_bgr.shape[:2]
    px, py = manual_seed if manual_seed is not None else _find_conjunctiva_seed(img_bgr)

    # Point positif = seed sur la conjonctive
    # Points négatifs = sclérotique (blanc de l'œil) + bord haut de l'image
    # → empêche SAM2 de sélectionner tout le visage
    points = [[px, py]]
    labels = [1]

    eye_center = _locate_eye_center(img_bgr)
    if eye_center is not None:
        ex, ey = eye_center
        # Point négatif au centre de la sclérotique (au-dessus du seed)
        points.append([ex, max(0, ey - int(h * 0.05))])
        labels.append(0)
    # Point négatif en haut de l'image (arrière-plan / peau)
    points.append([w // 2, int(h * 0.05)])
    labels.append(0)

    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = _SAM_MODEL.predict(img_rgb, points=[points], labels=[labels], verbose=False)

        if not results or results[0].masks is None:
            return None, {"sam_reason": "no_mask"}

        masks = results[0].masks.data.cpu().numpy()
        if len(masks) == 0:
            return None, {"sam_reason": "empty_masks"}

        # Sélectionner le masque dont le centroïde est le plus proche du seed
        best, best_score = None, float("inf")
        for m in masks:
            fill_m = float(m.mean() * 100)
            if fill_m < 0.1 or fill_m > 55:
                continue
            ys, xs = np.where(m > 0.5)
            if len(xs) == 0:
                continue
            mask_w = float(xs.max() - xs.min() + 1)
            mask_h = float(ys.max() - ys.min() + 1)
            # La conjonctive est large (paysage) — rejeter si plus haute que large
            if mask_h > mask_w * 1.4:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            score = dist / max(w, h) - np.log1p(m.sum()) * 0.002
            if score < best_score:
                best_score, best = score, m

        if best is None:
            return None, {"sam_reason": "no_valid_mask"}

        fill = float(best.mean() * 100)
        mask_raw = (best > 0.5).astype(np.uint8) * 255

        # ── Post-traitement adaptatif ──
        # Étape 1 : soustraction sclérotique
        sclera_mask = _detect_sclera_mask(img_bgr)
        mask_post   = cv2.bitwise_and(mask_raw, cv2.bitwise_not(sclera_mask))

        # Étape 2 : filtre rose adaptatif (P40 de a* dans le masque = garde les 60% + rouges)
        lab_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        a_f = lab_f[:, :, 1]
        L_f = lab_f[:, :, 0]
        S_f = hsv_f[:, :, 1]
        V_f = hsv_f[:, :, 2]
        R_f = img_bgr[:, :, 2].astype(np.float32)
        G_f = img_bgr[:, :, 1].astype(np.float32)

        a_in = a_f[mask_post > 0]
        if len(a_in) > 100:
            a_thresh = max(126.0, float(np.percentile(a_in, 40)))
        else:
            a_thresh = 127.0

        rose = (a_f > a_thresh) & (L_f > 50) & (V_f > 40) & (S_f > 12) & (R_f > G_f * 1.01)
        mask_u8 = cv2.bitwise_and(mask_post, rose.astype(np.uint8) * 255)

        # Fallback ultra-minimal si résultat trop petit
        if int(mask_u8.sum() // 255) < 300:
            rose_v2 = (a_f > 125.5) & (L_f > 45) & (V_f > 35)
            mask_u8 = cv2.bitwise_and(mask_post, rose_v2.astype(np.uint8) * 255)

        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  k5, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k5, iterations=1)

        # Garder uniquement la composante connectée contenant le seed point
        # → élimine les blobs de peau rose/joue disjoints de la conjonctive
        n_cc, cc_map = cv2.connectedComponents(mask_u8, connectivity=8)
        seed_label = int(cc_map[py, px]) if n_cc > 1 else 0
        if seed_label > 0:
            mask_u8 = np.where(cc_map == seed_label, 255, 0).astype(np.uint8)
        elif n_cc > 1:
            # seed tombe hors masque (filtré) → garder la plus grande composante
            areas = [(cc_map == lbl).sum() for lbl in range(1, n_cc)]
            best_lbl = int(np.argmax(areas)) + 1
            mask_u8 = np.where(cc_map == best_lbl, 255, 0).astype(np.uint8)

        info = {
            "method":        "sam2",
            "sam_seed":      (px, py),
            "a_rose_thresh": round(a_thresh, 1),
            "roi_fill_pct":  round(fill, 2),
            "n_pixels":      int(mask_u8.sum() // 255),
            "search_box":    (0, 0, w, h),
        }

        roi_bgr = img_bgr.copy()
        roi_bgr[mask_u8 == 0] = 0
        if int(mask_u8.sum() // 255) > 10:
            lab_r  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            px_lab = lab_r[mask_u8 > 0]
            px_bgr = img_bgr[mask_u8 > 0]
            info["lab_a_mean"] = round(float(px_lab[:, 1].mean()), 1)
            info["rgb_R_mean"] = round(float(px_bgr[:, 2].mean()), 1)
            info["rgb_G_mean"] = round(float(px_bgr[:, 1].mean()), 1)

        return roi_bgr, mask_u8, info

    except Exception as e:
        return None, {"sam_reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. MediaPipe Face Mesh
# ─────────────────────────────────────────────────────────────────────────────

_MP_AVAIL = None   # None=pas encore testé, True=ok, False=indisponible

# Landmarks MediaPipe Face Mesh (478 pts avec refine_landmarks=True)
_MP_LEFT_EYE_ALL   = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                       173, 157, 158, 159, 160, 161, 246]
_MP_RIGHT_EYE_ALL  = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                       398, 384, 385, 386, 387, 388, 466]
_MP_LEFT_LOWER_LID = [33, 7, 163, 144, 145, 153, 154, 155, 133]
_MP_RIGHT_LOWER_LID= [263, 249, 390, 373, 374, 380, 381, 382, 362]
_MP_LEFT_IRIS      = [468, 469, 470, 471, 472]  # iris pts (refine)
_MP_RIGHT_IRIS     = [473, 474, 475, 476, 477]


_MP_LANDMARKER = None   # instance réutilisable FaceLandmarker

def _try_load_mediapipe() -> bool:
    global _MP_AVAIL, _MP_LANDMARKER
    if _MP_AVAIL is not None:
        return _MP_AVAIL
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks import python as mp_tasks

        model_path = Path(__file__).parent / "face_landmarker.task"
        if not model_path.exists():
            _MP_AVAIL = False
            return False

        base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=0.25,
            min_face_presence_confidence=0.25,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _MP_LANDMARKER = mp_vision.FaceLandmarker.create_from_options(opts)
        _MP_AVAIL = True
    except Exception:
        _MP_AVAIL = False
    return _MP_AVAIL


def _segment_with_mediapipe(
    img_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """
    Localise la conjonctive palpébrale inférieure via MediaPipe FaceLandmarker.

    Stratégie :
      1. FaceLandmarker détecte iris + paupière inférieure (478 landmarks)
      2. ROI = entre le bord bas de l'iris et la ligne des cils (± marge)
      3. Filtre couleur a* adaptatif dans cette ROI exacte

    Retourne (roi_bgr, mask, info) ou None si pas de visage détecté.
    """
    if not _try_load_mediapipe() or _MP_LANDMARKER is None:
        return None

    import mediapipe as mp

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res      = _MP_LANDMARKER.detect(mp_image)

    if not res.face_landmarks:
        return None   # pas de visage → fallback SAM2

    lm_list = res.face_landmarks[0]   # liste de NormalizedLandmark
    n_lm    = len(lm_list)

    def px(idx):
        l = lm_list[idx]
        return int(l.x * w), int(l.y * h)

    eyes = [
        {"all": _MP_LEFT_EYE_ALL,  "lower": _MP_LEFT_LOWER_LID,  "iris": _MP_LEFT_IRIS},
        {"all": _MP_RIGHT_EYE_ALL, "lower": _MP_RIGHT_LOWER_LID, "iris": _MP_RIGHT_IRIS},
    ]

    best_mask = None
    best_n    = 0
    best_seed = None
    best_meta = {}

    for eye in eyes:
        # Points de contour complet de l'œil
        eye_pts = [px(i) for i in eye["all"] if i < n_lm]
        if len(eye_pts) < 6:
            continue

        xs_e = [p[0] for p in eye_pts]
        ys_e = [p[1] for p in eye_pts]
        eye_w = max(xs_e) - min(xs_e)
        if eye_w < 8:
            continue

        # Bord inférieur de l'iris (= limite supérieure de la zone conjonctive visible)
        has_iris = all(i < n_lm for i in eye["iris"])
        if has_iris:
            iris_pts      = [px(i) for i in eye["iris"]]
            iris_bottom_y = max(p[1] for p in iris_pts)
            iris_cx       = int(np.mean([p[0] for p in iris_pts]))
        else:
            iris_cx       = int(np.mean(xs_e))
            iris_bottom_y = int(np.mean(ys_e)) + int(eye_w * 0.18)

        # Bord inférieur de la paupière inférieure (ligne des cils)
        lower_pts    = [px(i) for i in eye["lower"] if i < n_lm]
        lower_max_y  = max(p[1] for p in lower_pts) if lower_pts else iris_bottom_y + int(eye_w * 0.25)

        # ROI globale : de l'iris jusqu'à modérément sous la paupière
        margin_x = int(eye_w * 0.22)
        x0 = max(0, min(xs_e) - margin_x)
        x1 = min(w, max(xs_e) + margin_x)
        y0 = max(0, min(ys_e))
        y1 = min(h, lower_max_y + int(eye_w * 1.0))

        if y1 <= y0 or x1 <= x0:
            continue

        roi = img_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        # Bande anatomique : la conjonctive est ENTRE l'iris et le bord de paupière
        # On restreint le filtre couleur à cette bande pour exclure la joue
        strip_y0 = max(0, iris_bottom_y - int(eye_w * 0.12) - y0)
        strip_y1 = min(roi.shape[0], lower_max_y + int(eye_w * 0.40) - y0)
        strip_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        if strip_y1 > strip_y0:
            strip_mask[strip_y0:strip_y1, :] = 255

        # Filtre couleur adaptatif dans la bande anatomique
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        a  = roi_lab[:, :, 1]
        L  = roi_lab[:, :, 0]
        S  = roi_hsv[:, :, 1]
        V  = roi_hsv[:, :, 2]
        R  = roi[:, :, 2].astype(np.float32)
        G  = roi[:, :, 1].astype(np.float32)

        # Seuil a* adaptatif P35 sur pixels non-cils dans la bande
        in_strip = strip_mask > 0
        valid = in_strip & (L > 45) & (L < 215) & (S > 8)
        a_vals   = a[valid]
        a_thresh = max(127.5, float(np.percentile(a_vals, 35)) if len(a_vals) > 50 else 129.0)

        roi_mask = (
            (a > a_thresh) &
            (L > 45) &
            (L < 220) &
            (R > G * 1.02) &
            (V > 35) &
            (S > 8)
        ).astype(np.uint8) * 255

        # Restreindre à la bande anatomique
        roi_mask = cv2.bitwise_and(roi_mask, strip_mask)

        # Ouvrir (supprime cils fins) puis fermer (bouche les trous)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN,  k3, iterations=1)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, k5, iterations=2)

        n_px = int(roi_mask.sum() // 255)
        if n_px > best_n:
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y0:y1, x0:x1] = roi_mask

            ys_m, xs_m = np.where(roi_mask > 0)
            seed = (x0 + int(xs_m.mean()), y0 + int(ys_m.mean())) if len(xs_m) > 0 \
                   else (iris_cx, (y0 + y1) // 2)

            best_n    = n_px
            best_mask = full_mask
            best_seed = seed
            best_meta = {"a_thresh": round(a_thresh, 1),
                         "box": (x0, y0, x1, y1)}

    if best_mask is None or best_n < 50:
        return None

    # Retirer la sclérotique
    sclera_mask = _detect_sclera_mask(img_bgr)
    best_mask   = cv2.bitwise_and(best_mask, cv2.bitwise_not(sclera_mask))

    # Garder la composante la plus proche du seed
    best_mask = _keep_closest_to_seed(best_mask, best_seed[0], best_seed[1])
    best_n    = int(best_mask.sum() // 255)

    if best_n < 50:
        return None

    roi_bgr  = img_bgr.copy()
    roi_bgr[best_mask == 0] = 0
    fill_pct = float(best_mask.mean() / 255 * 100)

    x0, y0, x1, y1 = best_meta["box"]
    info = {
        "method":        "mediapipe",
        "sam_seed":      best_seed,        # réutilise clé existante pour l'affichage debug
        "a_rose_thresh": best_meta["a_thresh"],
        "roi_fill_pct":  round(fill_pct, 2),
        "n_pixels":      best_n,
        "search_box":    (x0, y0, x1 - x0, y1 - y0),
    }
    if best_n > 10:
        px_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)[best_mask > 0]
        px_bgr = img_bgr[best_mask > 0]
        info["lab_a_mean"] = round(float(px_lab[:, 1].mean()), 1)
        info["rgb_R_mean"] = round(float(px_bgr[:, 2].mean()), 1)
        info["rgb_G_mean"] = round(float(px_bgr[:, 1].mean()), 1)

    return roi_bgr, best_mask, info


# ─────────────────────────────────────────────────────────────────────────────
# 3. U-Net (fallback)
# ─────────────────────────────────────────────────────────────────────────────

_UNET_PATH     = Path("/Users/cisseissa/Downloads/anemia_v5/unet_conjonctiva.pt")
_UNET_LOADED   = False
_predict_unet_fn = None


def _try_load_unet():
    global _UNET_LOADED, _predict_unet_fn
    if _UNET_LOADED:
        return _predict_unet_fn is not None
    if not _UNET_PATH.exists():
        _UNET_LOADED = True
        return False
    try:
        unet_root = str(_UNET_PATH.parent)
        if unet_root not in sys.path:
            sys.path.insert(0, unet_root)
        from segmentation_unet import predict_conjonctiva_mask
        _predict_unet_fn = lambda img: predict_conjonctiva_mask(img, _UNET_PATH)
        _UNET_LOADED = True
        return True
    except Exception:
        _UNET_LOADED = True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def segment_conjunctiva(
    img_bgr: np.ndarray,
    manual_seed: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Segmente la conjonctive palpébrale depuis un gros plan d'œil.

    Cascade : SAM2-Précis → YOLOv8 → MediaPipe → SAM2-simple → U-Net → heuristique.

    SAM2-Précis = multi-prompts anatomiques (5–8 positifs + 5–7 négatifs)
    avec sélection du meilleur masque et raffinement Lab.
    """
    h, w = img_bgr.shape[:2]

    # ── Méthode 0 : U-Net ResNet50 CIV (priorité maximale — test Dice=0.805) ─────
    r50 = _segment_with_best_unet(img_bgr)
    if r50 is not None:
        return r50

    # ── Méthode 1 : YOLOv8-seg dédié conjonctive (si USE_YOLO=True) ─────────────
    if USE_YOLO:
        try:
            import importlib.util as _ilu
            _yolo_file = Path(__file__).parent.parent / "segmentation" / "yolo_seg.py"
            _spec = _ilu.spec_from_file_location("_yolo_seg_mod", _yolo_file)
            _yolo_mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_yolo_mod)
            if _yolo_mod.is_available():
                mask_yolo, info_yolo = _yolo_mod.segment_conjunctiva_with_info(img_bgr)
                if mask_yolo is not None and mask_yolo.sum() > 0:
                    roi_bgr = img_bgr.copy()
                    roi_bgr[mask_yolo == 0] = 0
                    info_yolo["method"] = "yolo_seg"
                    return roi_bgr, mask_yolo, info_yolo
        except Exception as _e:
            import sys as _sys
            print(f"[YOLO_CASCADE_ERR] {type(_e).__name__}: {_e}", file=_sys.stderr)

    # ── Méthode 1 : SAM2 multi-prompts ultra-précis ───────────────────────────
    if _try_load_sam() and _SAM_MODEL is not None and manual_seed is None:
        try:
            from demo.conjunctiva_precise import segment_conjunctiva_precise
            mask_p, info_p = segment_conjunctiva_precise(img_bgr, _SAM_MODEL)
            n_p = info_p.get("n_pixels", 0)
            a_p = info_p.get("a_star_mean", 0)
            if n_p > 500 and a_p > 2.0:
                roi_bgr = img_bgr.copy()
                roi_bgr[mask_p == 0] = 0
                return roi_bgr, mask_p, info_p
        except Exception:
            pass

    # ── Méthode 2 : MediaPipe Face Mesh ──────────────────────────────────────
    # Seed forcé → skip MediaPipe et laisser SAM2 utiliser le point corrigé
    if manual_seed is None:
        mp_result = _segment_with_mediapipe(img_bgr)
        if mp_result is not None:
            roi_bgr, mask_u8, info = mp_result
            if info.get("n_pixels", 0) > 150:
                return roi_bgr, mask_u8, info

    # ── Méthode 2 : SAM2 ─────────────────────────────────────────────────────
    sam_result = _segment_with_sam(img_bgr, manual_seed=manual_seed)
    if sam_result is not None and len(sam_result) == 3:
        roi_bgr, mask_u8, info = sam_result
        if mask_u8 is not None and info.get("n_pixels", 0) > 100:
            return roi_bgr, mask_u8, info

    # ── Méthode 3 : U-Net ────────────────────────────────────────────────────
    if _try_load_unet() and _predict_unet_fn is not None:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_bool, conf = _predict_unet_fn(img_rgb)
            n_px  = int(mask_bool.sum()) if mask_bool is not None else 0
            fill  = float(n_px / (h * w) * 100)
            if mask_bool is not None and conf >= 0.20 and fill >= 0.3:
                mask_u8 = mask_bool.astype(np.uint8) * 255
                k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k5)
                roi_bgr = img_bgr.copy()
                roi_bgr[mask_u8 == 0] = 0
                info: dict = {
                    "method": "unet",
                    "unet_conf": round(conf, 3),
                    "roi_fill_pct": round(fill, 2),
                    "n_pixels": int(mask_u8.sum() // 255),
                    "search_box": (0, 0, w, h),
                }
                if n_px > 10:
                    lab_f  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
                    px_lab = lab_f[mask_u8 > 0]
                    px_bgr = img_bgr[mask_u8 > 0]
                    info["lab_a_mean"] = round(float(px_lab[:, 1].mean()), 1)
                    info["rgb_R_mean"] = round(float(px_bgr[:, 2].mean()), 1)
                    info["rgb_G_mean"] = round(float(px_bgr[:, 1].mean()), 1)
                return roi_bgr, mask_u8, info
        except Exception:
            pass

    # ── Méthode 4 : heuristique couleur ──────────────────────────────────────
    return _segment_heuristic(img_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristique couleur (dernier recours)
# ─────────────────────────────────────────────────────────────────────────────

def _segment_heuristic(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = img_bgr.shape[:2]

    # Seed anatomique — guide la sélection du bon composant
    sx, sy = _find_conjunctiva_seed(img_bgr)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, a = lab[:,:,0], lab[:,:,1]
    S, V = hsv[:,:,1], hsv[:,:,2]
    R = img_bgr[:,:,2].astype(np.float32)
    G = img_bgr[:,:,1].astype(np.float32)
    B = img_bgr[:,:,0].astype(np.float32)

    # ── Zone de recherche : ±40% x, ±28% y autour du seed ───────────────────
    zone = np.zeros((h, w), dtype=np.uint8)
    zx0 = max(0, sx - int(w * 0.40)); zx1 = min(w, sx + int(w * 0.40))
    zy0 = max(0, sy - int(h * 0.22)); zy1 = min(h, sy + int(h * 0.22))
    zone[zy0:zy1, zx0:zx1] = 255

    # ── Exclusion sclérotique (blanche ou jaunâtre) ──────────────────────────
    sclera = ((L > 150) & (S < 75) & (np.abs(a - 128) < 20)).astype(np.uint8) * 255
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    sclera_dil = cv2.dilate(sclera, k11, iterations=3)

    # ── Masque conjonctive : rouge, saturé, R >> G et R >> B ─────────────────
    # Seuil a* adaptatif calculé dans la zone seed uniquement
    a_zone = a[zy0:zy1, zx0:zx1]
    S_zone = S[zy0:zy1, zx0:zx1]
    L_zone = L[zy0:zy1, zx0:zx1]
    # Seuils abaissés pour peau sombre (L peut être < 55 sur conjonctive sombre)
    valid_zone = (L_zone > 30) & (L_zone < 220) & (S_zone > 10)
    a_valid = a_zone[valid_zone]
    # p70 au lieu de p80 : moins agressif quand la zone seed est imparfaite
    a_thresh = max(133.0, float(np.percentile(a_valid, 70)) if a_valid.size > 30 else 136.0)

    mask_full = (
        (a > a_thresh) &
        (L > 28) & (L < 215) &
        (R > G * 1.04) &
        (R > B * 1.05) &
        (S > 12) &
        (V > 22)
    ).astype(np.uint8) * 255

    mask_full = cv2.bitwise_and(mask_full, cv2.bitwise_not(sclera_dil))
    mask_full = cv2.bitwise_and(mask_full, zone)  # restreindre à la zone seed

    # Morphologie légère
    k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN,  k5,  iterations=1)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, k7h, iterations=1)

    # Garder le composant LE PLUS PROCHE DU SEED (pas le plus central)
    mask_full = _keep_closest_to_seed(mask_full, sx, sy)

    # Retry ultra-permissif si résultat insuffisant (peau très sombre, éclairage faible)
    if int(mask_full.sum() // 255) < 500:
        mask_permissive = (
            (a > 130.0) &
            (L > 18) & (L < 220) &
            (R > G * 1.02) &
            (S > 8) &
            (V > 15)
        ).astype(np.uint8) * 255
        mask_permissive = cv2.bitwise_and(mask_permissive, zone)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_permissive = cv2.morphologyEx(mask_permissive, cv2.MORPH_CLOSE, k5, iterations=2)
        mask_permissive = _keep_closest_to_seed(mask_permissive, sx, sy)
        if int(mask_permissive.sum() // 255) > int(mask_full.sum() // 255):
            mask_full = mask_permissive

    roi_bgr = img_bgr.copy()
    roi_bgr[mask_full == 0] = 0

    fill_pct = float(mask_full.mean() / 255 * 100)
    px = img_bgr[mask_full > 0]
    info: dict = {
        "method":       "adaptive",
        "a_thresh":     round(a_thresh, 1),
        "roi_fill_pct": round(fill_pct, 2),
        "n_pixels":     int(mask_full.sum() // 255),
        "search_box":   (0, 0, w, h),
    }
    if len(px) > 10:
        lab_full = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        px_lab   = lab_full[mask_full > 0]
        info["lab_a_mean"] = round(float(px_lab[:, 1].mean()), 1)
        info["rgb_R_mean"] = round(float(px[:, 2].mean()), 1)
        info["rgb_G_mean"] = round(float(px[:, 1].mean()), 1)

    return roi_bgr, mask_full, info


def _adaptive_conjunctiva_mask(
    region: np.ndarray,
    permissive: bool = False,
) -> tuple[np.ndarray, dict]:
    if region.size == 0:
        return np.zeros(region.shape[:2], dtype=np.uint8), {"method": "empty"}

    rh, rw = region.shape[:2]
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0].astype(np.float32)
    a   = lab[:, :, 1].astype(np.float32)
    bgr = region.astype(np.float32)
    R   = bgr[:, :, 2]
    G   = bgr[:, :, 1]

    inner_y0 = int(rh * 0.20); inner_y1 = int(rh * 0.80)
    inner_x0 = int(rw * 0.10); inner_x1 = int(rw * 0.90)
    a_inner  = a[inner_y0:inner_y1, inner_x0:inner_x1]

    pct          = 62 if permissive else 75
    a_percentile = float(np.percentile(a_inner, pct))
    a_cap        = 132.0 if permissive else 134.0
    a_thresh     = max(127.0 if permissive else 128.5, min(a_percentile, a_cap))

    l_min    = 25.0 if permissive else 35.0   # peau sombre : L peut être < 52
    r_min    = 40.0 if permissive else 55.0
    rg_ratio = 1.02 if permissive else 1.05

    mask = (
        (a   > a_thresh)
        & (L > l_min)
        & (L < 210.0)
        & (R  > G * rg_ratio)
        & (R  > r_min)
    ).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)

    stats = {
        "method":      "adaptive" + ("+permissive" if permissive else ""),
        "a_thresh":    round(a_thresh, 1),
        "a_p75_inner": round(a_percentile, 1),
    }
    return mask, stats


def _keep_closest_to_seed(mask: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """Garde le composant connecté le plus proche du point seed."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask

    best_label, best_score = -1, float("inf")
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 100:
            continue
        cx, cy = centroids[lbl]
        dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        # Favorise légèrement les grandes surfaces sans trop pénaliser la distance
        score = dist - float(np.log1p(area)) * 3.0
        if score < best_score:
            best_score = score
            best_label = lbl

    if best_label < 0:
        return mask
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Détection précise de la sclérotique (pour soustraction du masque SAM)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_sclera_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Détecte les pixels de sclérotique dans toute l'image.
    Retourne un masque uint8 (H,W) à 255 là où c'est de la sclérotique.

    Stratégie : la sclérotique est blanche et quasi-achromatique.
      - HSV : saturation S < 40 (peu de couleur) ET valeur V > 140 (lumineux)
      - Lab : L > 155 ET |a*-128| < 15 ET |b*-128| < 20
    On prend l'intersection des deux critères pour être précis.
    """
    # Critère 1 — HSV : sclérotique = peu saturée et lumineuse
    # S < 55 (élargi depuis 40) : sclérotique jaunâtre a S jusqu'à ~50
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    S   = hsv[:, :, 1]
    V   = hsv[:, :, 2]
    crit_hsv = (S < 55) & (V > 135)

    # Critère 2 — Lab : très lumineux, peu de rouge (a* neutre)
    # b* élargi à 35 : sclérotique jaunâtre chez peau foncée a b*~145-160
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L    = lab[:, :, 0]
    a    = lab[:, :, 1]
    b    = lab[:, :, 2]
    crit_lab = (L > 148) & (np.abs(a - 128) < 14) & (np.abs(b - 128) < 38)

    # Union des deux critères (OR) : attrape plus de sclérotique
    sclera = ((crit_hsv & (L > 148)) | crit_lab).astype(np.uint8) * 255

    # Fermeture morphologique (k7×2 : moins agressive que k9×3 pour éviter de rogner la conjonctive)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_CLOSE, k7, iterations=2)

    return sclera


# ─────────────────────────────────────────────────────────────────────────────
# Sclérotique + balance des blancs (inchangé)
# ─────────────────────────────────────────────────────────────────────────────

def detect_sclera(img_bgr: np.ndarray) -> dict | None:
    h, w = img_bgr.shape[:2]
    y0, y1 = int(h * 0.05), int(h * 0.45)
    region = img_bgr[y0:y1, :]

    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    bgr_f   = region.astype(np.float32)

    mask = (L > 175) & (np.abs(a - 128) < 12) & (np.abs(b - 128) < 18)
    n_px = int(mask.sum())
    if n_px < 50:
        return None

    return {
        "R_mean": float(bgr_f[:, :, 2][mask].mean()),
        "G_mean": float(bgr_f[:, :, 1][mask].mean()),
        "B_mean": float(bgr_f[:, :, 0][mask].mean()),
        "L_mean": float(L[mask].mean()),
        "a_mean": float(a[mask].mean()),
        "n_px":   n_px,
    }


def apply_sclera_wb(roi_bgr: np.ndarray, sclera: dict) -> np.ndarray:
    target = 220.0
    out    = roi_bgr.astype(np.float32)
    for ch_idx, key in enumerate(["B_mean", "G_mean", "R_mean"]):
        sc_val = sclera[key]
        if sc_val > 10:
            gain = target / sc_val
            out[:, :, ch_idx] = np.clip(out[:, :, ch_idx] * gain, 0, 255)
    return out.astype(np.uint8)


def draw_debug(img_bgr: np.ndarray, mask: np.ndarray, info: dict) -> np.ndarray:
    vis     = img_bgr.copy()
    overlay = img_bgr.copy()
    overlay[mask > 0] = (0, 200, 80)
    vis = cv2.addWeighted(overlay, 0.42, vis, 0.58, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 60), 2)

    x, y, bw, bh = info.get("search_box", (0, 0, 0, 0))
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 180, 255), 1)

    # Point seed SAM2 (point positif)
    if "sam_seed" in info:
        sx, sy = info["sam_seed"]
        cv2.circle(vis, (int(sx), int(sy)), 7, (0, 255, 0), -1)
        cv2.circle(vis, (int(sx), int(sy)), 9, (0, 200, 0), 2)

    h_img  = img_bgr.shape[0]
    a_str  = info.get("lab_a_mean", "?")
    r_str  = info.get("rgb_R_mean", "?")
    a_rose = info.get("a_rose_thresh", "")
    rose_s = f" ar={a_rose}" if a_rose else ""
    txt1   = f"fill={info['roi_fill_pct']:.1f}%  a*={a_str}  R={r_str}  [{info.get('method','?')}]{rose_s}"
    cv2.putText(vis, txt1, (8, h_img - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
