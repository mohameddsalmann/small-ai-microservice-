# Beard Classification Microservice

Small AI microservice that classifies whether a face in an image has a beard or not. Built with **FastAPI** and ready for **RunPod Serverless**.

## Features

- **Camera UI**: Open the app in a browser, allow camera access, and see live "Beard: Yes / No" on screen.
- **REST API**: `POST /classify` with image (file upload or JSON with `image_base64` / `image_url`).
- **RunPod Serverless**: Use the same logic via a RunPod handler; connect this repo to RunPod and deploy.

## Quick start (local)

```bash
# Create venv and install
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, allow camera, and click **Start camera**. The overlay will show "Beard: Yes" or "Beard: No" based on the current frame.

## API

### `POST /classify`

- **Multipart**: send a file with key `file`.
- **JSON**: `{ "image_base64": "<base64 string>" }` or `{ "image_url": "<url>" }`.

Response:

```json
{
  "has_beard": true,
  "confidence": 0.87,
  "message": "Beard detected"
}
```

### `GET /health`

Returns `{ "status": "ok", "service": "beard-classifier" }`.

## RunPod Serverless

1. **Connect GitHub**: In RunPod Console → Serverless → Deploy from GitHub, connect this repo.
2. **Build**: Use the repo’s default `Dockerfile`. It runs `runpod_handler/handler.py`.
3. **Input**: Send jobs with input:
   - `image_base64`: base64-encoded image string, or
   - `image_url`: URL of the image.
4. **Output**: Same shape as `/classify`: `has_beard`, `confidence`, `message`.

Optional: set env `BEARD_MODEL_PATH` to the path of your trained model file (e.g. `models/beard.pt`) inside the image if you add one.

## Model

The app uses:

- **Face detection**: OpenCV Haar cascade.
- **Beard classification**: Small MobileNetV3-based binary classifier (no_beard / beard). By default it uses ImageNet-pretrained backbone and an untrained head, so results are indicative. For production, train on a dataset with beard labels (e.g. CelebA “No_Beard”) and save weights to a `.pt` file, then load via `BEARD_MODEL_PATH` or by placing the file where your code expects it.

## Project layout

```
.
├── app/
│   ├── main.py          # FastAPI app and /classify
│   └── classifier.py    # Face detection + beard classifier
├── runpod_handler/
│   └── handler.py      # RunPod serverless handler
├── templates/
│   └── camera.html     # Camera UI
├── requirements.txt
├── Dockerfile          # RunPod serverless image
├── Dockerfile.api      # Optional: FastAPI-only image
└── README.md
```

## License

MIT.
