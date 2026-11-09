"""Web demo: uvicorn server:app --port 8000, then open http://localhost:8000

POST /api/predict  multipart field `file` (photo of the eye, lower eyelid pulled down)
                   -> {ok, probability, a_norm, triage, overlay (JPEG base64), debug} or {ok: false, error}
GET  /api/health
Images are processed in memory and never written to disk.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from anemia_eye import AnemiaEye

MAX_BYTES = 15 * 1024 * 1024
app = FastAPI(title='Yako anemia eye screening')
eye = AnemiaEye()


@app.get('/api/health')
def health():
    return {'ok': True}


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image too large (15 MB max)')
    r = eye.predict(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
    if not r.ok:
        return {'ok': False, 'error': r.error}
    _, jpg = cv2.imencode('.jpg', r.overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return {'ok': True, 'probability': r.probability, 'a_norm': r.a_norm, 'triage': r.triage,
            'overlay': base64.b64encode(jpg).decode(), 'debug': r.debug}


@app.get('/')
def index():
    return FileResponse(Path(__file__).parent / 'static' / 'index.html')
