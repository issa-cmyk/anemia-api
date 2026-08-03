"""
FastAPI backend — Triage Anémie
Endpoints consommés par le frontend React (Lovable / triage.africa)

Deploy: Railway / Render / Fly.io
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

# ── Chemins modèles (relatifs au dossier du script → fonctionne en local et sur Railway) ──
_HERE     = Path(__file__).parent
MODEL_AL  = _HERE / "model_aL_demo.pkl"

EPS             = 1e-6
NAIL_PALE_THR   = 0.966
CONJ_HB_THR     = 12.0

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Triage Anémie API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://triage.africa", "http://localhost:*", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement modèles (au démarrage) ─────────────────────────────────────────
try:
    conj_model = joblib.load(MODEL_AL)
except Exception:
    conj_model = None

try:
    from segmentation import segment_conjunctiva, check_conjunctiva_visibility
    HAS_SEG = True
except Exception:
    HAS_SEG = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_from_bytes(data: bytes) -> np.ndarray:
    from PIL import ImageOps
    pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def extract_nail_features(img_bgr, poly):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 1)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    spec = (hsv[:,:,1] < 30) & (hsv[:,:,2] > 220)
    valid = (mask == 1) & (~spec)
    if valid.sum() < 30:
        return None
    R   = img_bgr[:,:,2].astype(float)[valid]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
    L   = lab[:,:,0][valid] / 255 * 100
    a   = lab[:,:,1][valid] - 128
    pale     = float((R > 180).mean())
    pct_spec = float(((mask==1) & spec).sum()) / (int((mask==1).sum()) + 1)
    return {"pale": pale, "L": float(L.mean()), "a": float(a.mean()),
            "npx": int(valid.sum()), "pct_spec": pct_spec}

def extract_conj_from_mask(img_bgr, mask):
    valid = mask > 0
    if valid.sum() < 30:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(float)
    L   = lab[:,:,0][valid] / 255 * 100
    a   = lab[:,:,1][valid] - 128
    a_mean, L_mean = float(a.mean()), float(L.mean())
    return {"aL": a_mean/(L_mean+EPS), "a": a_mean, "L": L_mean, "npx": int(valid.sum())}

def mask_to_png_b64(mask: np.ndarray) -> str:
    import base64
    _, buf = cv2.imencode(".png", mask)
    return base64.b64encode(buf).decode()

def overlay_to_b64(img_bgr, mask, color=(0,200,80), alpha=0.35) -> str:
    import base64
    ov = img_bgr.copy()
    ov[mask > 0] = color
    vis = cv2.addWeighted(img_bgr, 1-alpha, ov, alpha, 0)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(vis_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + __import__("base64").b64encode(buf.getvalue()).decode()


# ── Endpoint santé ────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "conj_model": conj_model is not None, "seg": HAS_SEG}


# ── Endpoint ongles ───────────────────────────────────────────────────────────
@app.post("/api/nail/analyze")
async def analyze_nail(
    image: UploadFile = File(...),
    polygon: str = Form(...),          # JSON: [[x,y], [x,y], ...]  coords dans l'image originale
):
    """
    Analyse le lit unguéal.
    polygon: coordonnées dans l'image originale (pas affichage).
    """
    data = await image.read()
    try:
        img_bgr = pil_from_bytes(data)
    except Exception:
        raise HTTPException(400, "Image illisible")

    try:
        poly = json.loads(polygon)
        if len(poly) < 3:
            raise ValueError()
    except Exception:
        raise HTTPException(400, "polygon invalide (JSON [[x,y],...], min 3 points)")

    feats = extract_nail_features(img_bgr, poly)
    if feats is None:
        raise HTTPException(422, "Zone trop petite ou entièrement spéculaire")

    pale      = feats["pale"]
    high_risk = pale >= NAIL_PALE_THR

    # Overlay visuel
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    overlay_b64 = overlay_to_b64(img_bgr, mask, color=(0,200,80))

    return {
        "high_risk":   high_risk,
        "pale":        round(pale, 4),
        "L":           round(feats["L"], 1),
        "a":           round(feats["a"], 2),
        "npx":         feats["npx"],
        "pct_spec":    round(feats["pct_spec"], 3),
        "threshold":   NAIL_PALE_THR,
        "overlay_img": overlay_b64,
    }


# ── Endpoint conjonctive ──────────────────────────────────────────────────────
@app.post("/api/conj/analyze")
async def analyze_conj(
    image: UploadFile = File(...),
    polygon: Optional[str] = Form(None),  # JSON [[x,y],...] OU null → auto-seg
):
    """
    Analyse la conjonctive palpébrale.
    Si polygon=null et HAS_SEG=True → segmentation auto U-Net.
    Sinon polygon requis (coords image originale).
    """
    data = await image.read()
    try:
        img_bgr = pil_from_bytes(data)
    except Exception:
        raise HTTPException(400, "Image illisible")

    h, w = img_bgr.shape[:2]
    method = "manual"

    if polygon:
        try:
            poly = json.loads(polygon)
            if len(poly) < 3:
                raise ValueError()
        except Exception:
            raise HTTPException(400, "polygon invalide")
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    elif HAS_SEG:
        try:
            _, mask, info = segment_conjunctiva(img_bgr)
            method = info.get("method", "auto")
        except Exception:
            raise HTTPException(500, "Segmentation automatique échouée — envoyer polygon")
        if int(mask.sum() // 255) < 300:
            raise HTTPException(422, "Conjonctive non détectée — annoter manuellement")
    else:
        raise HTTPException(422, "Segmentation auto non disponible — envoyer polygon")

    feats = extract_conj_from_mask(img_bgr, mask)
    if feats is None:
        raise HTTPException(422, "Zone trop petite")

    aL      = feats["aL"]
    hb_pred = None
    if conj_model is not None:
        try:
            hb_pred = float(conj_model.predict([[aL]])[0])
        except Exception:
            pass

    high_risk = hb_pred is not None and hb_pred < CONJ_HB_THR
    overlay_b64 = overlay_to_b64(img_bgr, mask, color=(0,200,80))

    return {
        "high_risk":   high_risk,
        "hb_pred":     round(hb_pred, 1) if hb_pred is not None else None,
        "aL":          round(aL, 4),
        "a":           round(feats["a"], 2),
        "L":           round(feats["L"], 1),
        "npx":         feats["npx"],
        "method":      method,
        "threshold_hb": CONJ_HB_THR,
        "overlay_img": overlay_b64,
    }


# ── Endpoint auto-seg preview ─────────────────────────────────────────────────
@app.post("/api/conj/segment")
async def segment_conj_preview(image: UploadFile = File(...)):
    """Retourne l'overlay de segmentation auto (avant confirmation)."""
    if not HAS_SEG:
        raise HTTPException(503, "Segmentation non disponible")

    data = await image.read()
    try:
        img_bgr = pil_from_bytes(data)
    except Exception:
        raise HTTPException(400, "Image illisible")

    try:
        _, mask, info = segment_conjunctiva(img_bgr)
    except Exception:
        raise HTTPException(500, "Erreur segmentation")

    n_px = int(mask.sum() // 255)
    if n_px < 300:
        return {"success": False, "npx": n_px, "method": "failed", "overlay_img": None}

    overlay_b64 = overlay_to_b64(img_bgr, mask, color=(0,200,80))
    return {
        "success":     True,
        "npx":         n_px,
        "method":      info.get("method", "auto"),
        "overlay_img": overlay_b64,
    }
