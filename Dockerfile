FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Télécharge les modèles depuis Hugging Face
RUN wget -q -O /app/best_unet.pt \
    "https://huggingface.co/pk180/anemia-models/resolve/main/best_unet.pt"

RUN wget -q -O /app/model_aL_demo.pkl \
    "https://huggingface.co/pk180/anemia-models/resolve/main/model_aL_demo.pkl"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
