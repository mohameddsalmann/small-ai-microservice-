"""
Beard classification microservice - FastAPI app.
Serves camera UI and /classify endpoint for RunPod-compatible API.
"""
import base64
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.classifier import BeardClassifier

app = FastAPI(
    title="Beard Classification API",
    description="Classify if a face in an image has a beard or not",
    version="1.0.0",
)

# Load classifier once at startup (RunPod best practice: init outside handler)
classifier = BeardClassifier()

# Serve static files if any
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class ClassifyResponse(BaseModel):
    has_beard: bool
    confidence: float
    message: str


class ClassifyRequest(BaseModel):
    """For JSON body with base64 image (RunPod / API clients)."""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def camera_page():
    """Serve the camera UI page."""
    html_path = Path(__file__).parent.parent / "templates" / "camera.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="camera.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/classify", response_model=ClassifyResponse)
async def classify_image(request: Request, file: Optional[UploadFile] = File(None)):
    """
    Classify beard from image.
    Accepts: multipart file upload, or JSON body with image_base64 / image_url.
    """
    image_bytes = None
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        b64 = body.get("image_base64") if isinstance(body, dict) else None
        url = body.get("image_url") if isinstance(body, dict) else None
        if b64:
            try:
                image_bytes = base64.b64decode(b64)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
        elif url:
            import urllib.request
            try:
                with urllib.request.urlopen(url) as resp:
                    image_bytes = resp.read()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image URL: {e}")
    if image_bytes is None and file and file.filename:
        image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Provide image via file upload or JSON body (image_base64 or image_url)",
        )

    try:
        has_beard, confidence = classifier.predict(image_bytes)
        message = "Beard detected" if has_beard else "No beard detected"
        return ClassifyResponse(
            has_beard=has_beard,
            confidence=round(float(confidence), 4),
            message=message,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "beard-classifier"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
