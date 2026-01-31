"""
Beard classifier: face detection + binary classification.
Uses OpenCV for face detection and a small CNN for beard/no-beard.
Model can be replaced with your own trained weights (e.g. CelebA beard attribute).
"""
import os
from typing import Optional

import cv2
import numpy as np

# Optional PyTorch for neural classifier; fallback to heuristic if not available
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False

# Input size for the classifier
INPUT_SIZE = 224
# Default confidence threshold
BEARD_THRESHOLD = 0.5


def _get_face_cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OpenCV face cascade not found: {path}")
    return cv2.CascadeClassifier(path)


if TORCH_AVAILABLE:
    class BeardNet(nn.Module):
        """Small CNN for beard classification (2 classes: no_beard, beard)."""

        def __init__(self, num_classes=2):
            super().__init__()
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.backbone(x)
else:
    BeardNet = None


class BeardClassifier:
    """Detect face in image and classify beard presence."""

    def __init__(self, model_path: Optional[str] = None):
        self._face_cascade = _get_face_cascade()
        self._device = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self._model = None
        self._transform = None

        if TORCH_AVAILABLE:
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._model = BeardNet(num_classes=2)
            if model_path and os.path.isfile(model_path):
                self._model.load_state_dict(torch.load(model_path, map_location=self._device))
            self._model.to(self._device)
            self._model.eval()

    def _detect_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return image[y : y + h, x : x + w]

    def predict(self, image_bytes: bytes) -> tuple[bool, float]:
        """
        Predict beard from raw image bytes.
        Returns (has_beard: bool, confidence: float).
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")

        face = self._detect_face(image)
        if face is None:
            raise ValueError("No face detected in the image")

        if TORCH_AVAILABLE and self._model is not None:
            with torch.no_grad():
                face_bgr = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                tensor = self._transform(face_bgr).unsqueeze(0).to(self._device)
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)
                # index 0: no_beard, index 1: beard
                beard_prob = probs[0, 1].item()
            has_beard = beard_prob >= BEARD_THRESHOLD
            confidence = beard_prob if has_beard else (1.0 - beard_prob)
            return has_beard, confidence

        # Fallback: simple heuristic (lower-face variance / color) - placeholder only
        h, w = face.shape[:2]
        lower_face = face[int(h * 0.5) :, :]
        gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
        variance = float(np.var(gray_lower))
        # Arbitrary threshold for demo; replace with real model
        beard_prob = min(0.95, max(0.05, (variance - 200) / 800))
        has_beard = beard_prob >= BEARD_THRESHOLD
        confidence = beard_prob if has_beard else (1.0 - beard_prob)
        return has_beard, confidence
