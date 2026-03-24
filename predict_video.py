import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from face_cropper import crop_faces
from utils import enhance_image   # FIX 8: shared utility, not duplicated
import cv2
import numpy as np
import urllib.request

# ==========================
# MODEL DOWNLOAD
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_resnet18_best.pth")
URL = "https://huggingface.co/Navxx/DEEPFAKE-DETECTION1/resolve/main/deepfake_resnet18_best.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading deepfake model...")
    urllib.request.urlretrieve(URL, MODEL_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Human Real Face", "Human Fake Face"]

IMG_SIZE = 224
TTA_RUNS = 2

# ==========================
# MODEL ARCHITECTURE
# ==========================

model = resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully")
print("Using device:", DEVICE)

# ==========================
# TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# TTA
# ==========================

def predict_with_tta(face_img: np.ndarray) -> torch.Tensor:
    """
    Run TTA over the face image (RGB uint8, any size).
    Returns a 1D tensor of shape [2] with softmax probabilities.
    """
    probs = []
    for i in range(TTA_RUNS):
        img = face_img.copy()
        if i == 1:
            img = cv2.flip(img, 1)
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            probs.append(prob)
    return torch.mean(torch.stack(probs), dim=0).squeeze()

# ==========================
# BLUR CHECK
# ==========================

def laplacian_variance(rgb_image: np.ndarray) -> float:
    """Compute Laplacian variance on grayscale for a reliable blur score."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ==========================
# MAIN FUNCTION
# ==========================

def predict_image(image_path: str) -> dict:
    """
    Predict whether the face(s) in image_path are real or fake.

    Returns a dict with keys:
        label         — overall prediction string
        confidence    — 0-100 float
        faces         — list of cropped face paths
        boxes         — list of (x, y, w, h) tuples
        face_results  — list of (label, confidence) per face
    """
    cropped_paths, face_boxes = crop_faces(image_path)

    # ==========================
    # NO FACE DETECTED
    # ==========================
    if len(cropped_paths) == 0:
        image = cv2.imread(image_path)
        if image is None:
            return {
                "label": "Invalid Image",
                "confidence": 0.0,
                "faces": [],
                "boxes": [],
                "face_results": []
            }

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        if np.std(image) < 60:
            image = enhance_image(image)

        prob = predict_with_tta(image)
        fake_p = prob[1].item()
        real_p = prob[0].item()

        if fake_p >= real_p:
            label, confidence = "Human Fake Face", fake_p * 100
        else:
            label, confidence = "Human Real Face", real_p * 100

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "faces": [],
            "boxes": [],
            "face_results": []
        }

    # ==========================
    # FACE DETECTED
    # ==========================
    fake_probs = []
    face_results = []

    for path in cropped_paths:
        face = cv2.imread(path) if isinstance(path, str) else path
        if face is None:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.ascontiguousarray(face)

        # FIX 4: Check size BEFORE resizing — after resize it is always 224x224
        # so the check was dead code and tiny/noisy detections were never skipped.
        if face.shape[0] < 50 or face.shape[1] < 50:
            continue

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        if laplacian_variance(face) < 30:
            continue

        if np.std(face) < 60:
            face = enhance_image(face)

        prob = predict_with_tta(face)
        fake_p = prob[1].item()
        real_p = prob[0].item()

        if max(fake_p, real_p) < 0.50:
            continue

        fake_probs.append(fake_p)

        if fake_p >= real_p:
            label, conf = CLASS_NAMES[1], fake_p * 100
        else:
            label, conf = CLASS_NAMES[0], real_p * 100

        face_results.append((label, round(conf, 2)))

    # ==========================
    # AFTER LOOP
    # ==========================
    if not fake_probs:
        return {
            "label": "Low Quality / Uncertain",
            "confidence": 0.0,
            "faces": cropped_paths,
            "boxes": face_boxes,
            "face_results": []
        }

    mean_fake_prob = float(np.mean(fake_probs))

    print(f"Per-face fake probs: {[round(p, 3) for p in fake_probs]}")
    print(f"Mean fake prob: {mean_fake_prob:.3f}")

    FAKE_THRESHOLD = 0.55
    REAL_THRESHOLD = 0.45

    if mean_fake_prob >= FAKE_THRESHOLD:
        final_label = "Human Fake Face"
        final_confidence = mean_fake_prob * 100
    elif mean_fake_prob <= REAL_THRESHOLD:
        final_label = "Human Real Face"
        final_confidence = (1.0 - mean_fake_prob) * 100
    else:
        final_label = "Uncertain"
        final_confidence = abs(mean_fake_prob - 0.5) * 200

    return {
        "label": final_label,
        "confidence": round(min(final_confidence, 100.0), 2),
        "faces": cropped_paths,
        "boxes": face_boxes,
        "face_results": face_results
    }