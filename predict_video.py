import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
import cv2
import numpy as np

# ==========================
# SETTINGS (OPTIMIZED)
# ==========================

MODEL_PATH = "https://drive.google.com/uc?id=1-a2E3_hsSKm_BsxvZlT_dJ8oP3Z8J6FK"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1-a2E3_hsSKm_BsxvZlT_dJ8oP3Z8J6FK"
    gdown.download(url, MODEL_PATH, quiet=False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Real Video", "Fake Video"]

IMG_SIZE = 256
TTA_RUNS = 2
FRAME_SKIP = 5
FAKE_THRESHOLD = 0.55
MIN_FRAME_CONF = 0.7
MIN_FACE_SIZE = 80

# ==========================
# LOAD MODEL
# ==========================

model = resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

print("Model Loaded on:", DEVICE)

# ==========================
# TRANSFORM
# ==========================

base_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# FAST FACE DETECTOR (OpenCV DNN)
# ==========================

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ==========================
# LIGHT FACE ENHANCEMENT
# ==========================

def enhance_face(face):
    return cv2.GaussianBlur(face, (3, 3), 0)

# ==========================
# TTA PREDICTION (LIGHT)
# ==========================

def predict_face(face_img):
    probs = []

    for i in range(TTA_RUNS):
        img_variant = face_img.copy()

        if i == 1:
            img_variant = cv2.flip(img_variant, 1)

        tensor = base_transform(img_variant).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            probs.append(prob)

    mean_prob = torch.mean(torch.stack(probs), dim=0)

    return mean_prob[0][0].item(), mean_prob[0][1].item()

# ==========================
# FACE DETECTION FUNCTION
# ==========================

def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            w_box = x2 - x1
            h_box = y2 - y1

            if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
                continue

            faces.append((x1, y1, x2, y2))

    return faces

# ==========================
# MAIN
# ==========================

if len(sys.argv) < 2:
    print("Usage: python predict_video_fast.py <video_path>")
    sys.exit()

video_path = sys.argv[1]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video.")
    sys.exit()

frame_id = 0
fake_scores = []

print("\nAnalyzing:", video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    if frame_id % FRAME_SKIP != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detect_faces_dnn(frame)

    for (x1, y1, x2, y2) in faces:

        margin = int(0.2 * (x2 - x1))

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(rgb.shape[1], x2 + margin)
        y2 = min(rgb.shape[0], y2 + margin)

        face = rgb[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = enhance_face(face)

        real_p, fake_p = predict_face(face)

        if max(real_p, fake_p) < MIN_FRAME_CONF:
            continue

        fake_scores.append(fake_p)

cap.release()

# ==========================
# FINAL DECISION (STABLE)
# ==========================

if len(fake_scores) == 0:
    print("\nNo confident faces detected.")
    sys.exit()

fake_scores = np.array(fake_scores)

# Temporal smoothing
window = min(5, len(fake_scores))
smoothed = np.convolve(fake_scores,
                       np.ones(window) / window,
                       mode='valid')

# Stable decision using median
representative_score = np.median(smoothed)

# Ratio-based decision
fake_ratio = np.mean(fake_scores > 0.5)

is_fake = fake_ratio > 0.4
confidence = fake_ratio if is_fake else (1 - fake_ratio)

print("\n" + "=" * 40)
print("FINAL VIDEO RESULT")
print("=" * 40)

print("Prediction :", CLASS_NAMES[1] if is_fake else CLASS_NAMES[0])
print(f"Confidence : {confidence * 100:.2f}%")
print("Frames Analyzed :", len(fake_scores))
print("=" * 40)