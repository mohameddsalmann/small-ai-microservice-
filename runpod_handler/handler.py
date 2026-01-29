"""
RunPod serverless handler for beard classification.
Connect this repo to RunPod Serverless and use this handler.
Input: { "image_base64": "<base64 string>" } or { "image_url": "<url>" }
Output: { "has_beard": bool, "confidence": float, "message": str }
"""
import base64
import os
import sys

# Add project root so we can import app.classifier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runpod
from app.classifier import BeardClassifier

# Initialize model once outside handler (RunPod best practice)
MODEL_PATH = os.environ.get("BEARD_MODEL_PATH")
classifier = BeardClassifier(model_path=MODEL_PATH)


def handler(job):
    """
    RunPod job handler.
    job["input"] may contain: image_base64 (str), or image_url (str).
    """
    job_input = job.get("input", {})
    if not job_input:
        return {"error": "Input is required. Provide image_base64 or image_url."}

    image_base64 = job_input.get("image_base64")
    image_url = job_input.get("image_url")

    image_bytes = None
    if image_base64:
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return {"error": f"Invalid base64 image: {e}"}
    elif image_url:
        try:
            import urllib.request
            with urllib.request.urlopen(image_url) as resp:
                image_bytes = resp.read()
        except Exception as e:
            return {"error": f"Failed to fetch image URL: {e}"}
    else:
        return {"error": "Provide image_base64 or image_url in input."}

    if not image_bytes:
        return {"error": "Empty image."}

    try:
        has_beard, confidence = classifier.predict(image_bytes)
        message = "Beard detected" if has_beard else "No beard detected"
        return {
            "has_beard": has_beard,
            "confidence": round(float(confidence), 4),
            "message": message,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
