# anemia-api

Smartphone screening for **iron-deficiency anaemia** from a photo of the eye's conjunctiva (inner eyelid). Given an image, the API segments the conjunctiva, measures its colour, and returns a triage result — a low-cost, non-invasive flag for who most needs a confirmatory blood test.

Backend for the *Triage Anémie* workflow. FastAPI, served as a Docker container.

## Why
Anaemia is common and under-diagnosed where lab access is limited. A conjunctiva photo from any phone can help prioritise testing, without a needle.

## How it works
- **Segmentation:** the conjunctiva is isolated automatically with a **U-Net**, or from a **manual polygon** sent by the client (fallback when auto-segmentation fails).
- **Feature:** conjunctival colour in Lab space — the **a\*/L ratio** (redness normalised by lightness), which is more robust than raw redness across lighting and phone models.
- **Model:** maps the colour feature to a predicted haemoglobin; the API flags **high risk when predicted Hb < 12 g/dL**.
- **Also:** a nail-bed endpoint based on pallor, for comparison.

## Results (honest)
- **AUC ≈ 0.74** under **nested cross-validation** (N ≈ 100), on a cohort recruited on **prescribed blood tests** — not on appearance, to avoid verification bias.
- I traced and rejected a tempting AUC ≈ 0.80 once I found it came from data snooping.
- **Known limits:** colour-based signal has a ceiling; performance varies with skin phototype and device; severe-anaemia strata are small. This is a **screening aid, not a diagnosis**.

## API

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/` | Health check — model/segmentation status |
| `POST` | `/api/conj/analyze` | Analyse a conjunctiva image → triage result |
| `POST` | `/api/conj/segment` | Segmentation preview (overlay) |
| `POST` | `/api/nail/analyze` | Analyse a nail-bed image (pallor) |

`POST /api/conj/analyze` — multipart form:
- `image` (file, required)
- `polygon` (optional JSON `[[x,y],...]` in original-image coordinates). If omitted, the U-Net segments the conjunctiva automatically.

Response:
```json
{
  "high_risk": true,
  "hb_pred": 10.8,
  "aL": 1.0342,
  "a": 12.5,
  "L": 139.0,
  "method": "auto",
  "threshold_hb": 12.0,
  "overlay_img": "data:image/jpeg;base64,..."
}
```

## Run with Docker (recommended)
The Dockerfile fetches the model weights (U-Net + triage model) from Hugging Face at build time.
```bash
docker build -t anemia-api .
docker run -p 8000:8000 anemia-api
# → http://localhost:8000  (GET / should return {"status":"ok"})
```

## Run locally (without Docker)
Place the two model files next to `main.py` first: `best_unet.pt` and `model_aL_demo.pkl`.
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# POST an image to http://localhost:8000/api/conj/analyze
```

## Example request
```bash
curl -X POST http://localhost:8000/api/conj/analyze \
  -F "image=@eye.jpg"
```

## Stack
Python · FastAPI · PyTorch + segmentation-models-pytorch (U-Net) · OpenCV · scikit-learn

## Disclaimer
Research and screening prototype. Not a medical device and not a substitute for a blood test or a clinician's judgement.
