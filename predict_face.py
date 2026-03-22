import os
import gdown
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from face_cropper import crop_faces
import cv2
import numpy as np

# ==========================
# MODEL DOWNLOAD (FIXED)
# ==========================

MODEL_URL = "https://drive.google.com/uc?id=1-a2E3_hsSKm_BsxvZlT_dJ8oP3Z8J6FK"
MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ==========================
# SETTINGS
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Human Real Face", "Human Fake Face"]

IMG_SIZE = 256
TTA_RUNS = 2
FAKE_THRESHOLD = 0.75
USE_ENHANCEMENT = False

# ==========================
# LOAD MODEL
# ==========================

from torchvision.models import ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

# FREEZE BACKBONE
for param in model.parameters():
    param.requires_grad = False

# UNFREEZE FC
for param in model.fc.parameters():
    param.requires_grad = True

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
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================
# TTA
# ==========================

def predict_with_tta(face_img):

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

    mean_prob = torch.mean(torch.stack(probs), dim=0)

    return mean_prob.squeeze()

# ==========================
# MAIN FUNCTION
# ==========================

def predict_image(image_path):

    cropped_paths, face_boxes = crop_faces(image_path)

    if len(cropped_paths) == 0:
        return {
            "label": "No Face Detected",
            "confidence": 0.0,
            "faces": [],
            "boxes": [],
            "face_results": []
        }

    fake_probs = []
    face_results = []

    for path in cropped_paths:

        if isinstance(path, str):
            face = cv2.imread(path)
        else:
            face = path

        if face is None:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.ascontiguousarray(face)

        if face.shape[0] < 50 or face.shape[1] < 50:
            continue

        prob = predict_with_tta(face)

        fake_probability = prob[1].item()
        real_probability = prob[0].item()

        # STEP 3 (skip weak predictions)
        if abs(fake_probability - real_probability) < 0.15:
            continue

        if fake_probability < 0.3 and real_probability < 0.3:
            continue

        if fake_probability > real_probability:
            label = CLASS_NAMES[1]
            confidence = fake_probability * 100
        else:
            label = CLASS_NAMES[0]
            confidence = real_probability * 100

        fake_probs.append(fake_probability)
        face_results.append((label, confidence))

    if len(fake_probs) == 0:
        return {
            "label": "Uncertain",
            "confidence": 0.0,
            "faces": [],
            "boxes": [],
            "face_results": []
        }

    avg_fake_prob = max(fake_probs)

    if avg_fake_prob > FAKE_THRESHOLD:
        final_result = CLASS_NAMES[1]
        final_confidence = avg_fake_prob * 100
    else:
        final_result = CLASS_NAMES[0]
        final_confidence = (1 - avg_fake_prob) * 100

    final_confidence = max(final_confidence, 50)

    print("Fake probs:", fake_probs)
    print("Final:", avg_fake_prob)

    return {
        "label": final_result,
        "confidence": final_confidence,
        "faces": cropped_paths,
        "boxes": face_boxes,
        "face_results": face_results
    }