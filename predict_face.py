"""
predictor.py — Image deepfake predictor (ResNet18 backbone).

Loads the Navxx/DEEPFAKE-DETECTION1 ResNet18 checkpoint.
Class layout in this checkpoint:  index 0 = Real,  index 1 = Fake.

If predict_image() consistently returns wrong labels on images you KNOW
are real or fake, run diagnose.py first — it will tell you whether the
checkpoint has swapped class indices or a collapsed decision boundary.

REQUIREMENTS
============
    pip install torch torchvision opencv-python

Depends on:
    utils.py        — enhance_image, gentle_sharpen, laplacian_variance
    face_cropper.py — crop_faces  (returns 3-tuple)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from face_cropper import crop_faces
from utils import enhance_image, gentle_sharpen, laplacian_variance
import cv2
import numpy as np
import urllib.request
from typing import List, Tuple

# ==========================
# SETTINGS
# ==========================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_resnet18_best.pth")
MODEL_URL  = (
    "https://huggingface.co/Navxx/DEEPFAKE-DETECTION1"
    "/resolve/main/deepfake_resnet18_best.pth"
)

IMG_SIZE = 224

# Decision thresholds on fake_prob (softmax index 1).
# Uncertain band is deliberately narrow (0.45–0.55) so genuine
# borderline cases are the only ones left unclassified.
FAKE_THRESHOLD = 0.55   # ≥ this  → Fake
REAL_THRESHOLD = 0.45   # ≤ this  → Real
                        # between → Uncertain

# Quality gates
MIN_FACE_PX    = 40     # skip crops smaller than this in either dimension
BLUR_THRESHOLD = 40     # skip crops with Laplacian variance below this

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Human Real Face", "Human Fake Face"]


# ==========================
# MODEL DOWNLOAD & LOAD
# ==========================

def _download_if_missing(path: str, url: str, min_mb: float = 10.0):
    if os.path.exists(path):
        return
    print(f"Downloading model from {url} …")
    urllib.request.urlretrieve(url, path)
    size_mb = os.path.getsize(path) / 1e6
    if size_mb < min_mb:
        os.remove(path)
        raise RuntimeError(
            f"Downloaded file is only {size_mb:.1f} MB (expected ≥ {min_mb} MB).\n"
            "The URL likely returned an HTML error page instead of the binary.\n"
            f"Delete '{path}', fix the URL, or download the checkpoint manually."
        )
    print(f"✓ Model downloaded ({size_mb:.1f} MB)")


def _build_model() -> nn.Module:
    """Build the ResNet18 architecture that matches the Navxx checkpoint."""
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    return m


def _load_checkpoint(m: nn.Module, path: str) -> nn.Module:
    """
    Load checkpoint with automatic key-prefix stripping.

    Handles:
      - Raw OrderedDict of weights (most common)
      - Dicts wrapped under 'state_dict', 'model_state_dict', or 'model'
      - DataParallel 'module.' prefix

    Raises RuntimeError if > 50% of model keys are missing, which means
    the checkpoint architecture does not match the model definition.
    """
    raw = torch.load(path, map_location=DEVICE)

    # Unwrap common wrapper dicts
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            break

    if not isinstance(raw, dict):
        raise ValueError(
            f"Unexpected checkpoint format: {type(raw)}. "
            "Expected an OrderedDict of weight tensors."
        )

    # Strip DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in raw.items()}

    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [WARN] {len(missing)} missing key(s),  "
              f"{len(unexpected)} unexpected key(s).")
        if missing:
            print(f"         First missing : {missing[:3]}")
        if len(missing) > len(state) * 0.5:
            raise RuntimeError(
                "More than 50% of model keys are missing from the checkpoint.\n"
                "The checkpoint architecture does not match this model definition.\n"
                f"Model expects: {list(m.state_dict().keys())[:3]}\n"
                f"Checkpoint has: {list(state.keys())[:3]}\n"
                "Run diagnose.py to identify the exact problem."
            )
    else:
        print("✓ Checkpoint loaded — all keys matched.")
    return m


_download_if_missing(MODEL_PATH, MODEL_URL)

print("Loading deepfake detection model …")
model = _build_model()
model = _load_checkpoint(model, MODEL_PATH)
model = model.to(DEVICE)
model.eval()
print(f"✓ Predictor ready.  Device: {DEVICE}")


# ==========================
# TRANSFORM
# ==========================

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==========================
# TEST-TIME AUGMENTATION  (5 runs)
# ==========================

def _tta_augment(img: np.ndarray, run: int) -> np.ndarray:
    """Return an augmented copy for TTA run `run`."""
    out = img.copy()
    if run == 1:
        out = cv2.flip(out, 1)                              # horizontal mirror
    elif run == 2:
        out = cv2.convertScaleAbs(out, alpha=1.10, beta=10) # +brightness
    elif run == 3:
        out = cv2.convertScaleAbs(out, alpha=0.90, beta=-10)# -brightness
    elif run == 4:
        out = cv2.flip(out, 1)
        out = cv2.convertScaleAbs(out, alpha=1.05, beta=5)  # mirror + slight bright
    return out


TTA_RUNS = 5


def predict_with_tta(face_img: np.ndarray) -> torch.Tensor:
    """
    5-run TTA inference on a single face image (RGB uint8, any size).

    Returns a 1-D tensor [real_prob, fake_prob] averaged over all runs.
    Horizontal flip is always valid for faces; brightness shifts test
    robustness to exposure variation.
    """
    face_img = np.ascontiguousarray(face_img)
    probs = []
    for i in range(TTA_RUNS):
        aug    = _tta_augment(face_img, i)
        tensor = _transform(aug).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            p      = F.softmax(logits, dim=1)
            probs.append(p)
    return torch.mean(torch.stack(probs), dim=0).squeeze()   # [real_p, fake_p]


# ==========================
# GRAD-CAM  (visualisation only — does not affect predictions)
# ==========================

def get_gradcam(face_img: np.ndarray) -> np.ndarray:
    """
    GradCAM heatmap using ResNet18's last residual block (layer4).

    Highlights regions that pushed the model toward its prediction.
    Returns a float32 array in [0, 1] with the same H×W as face_img.
    Returns a zero array silently if anything goes wrong.

    NOTE: model.layer4 is an attribute of the ResNet BASE object that
    was wrapped — it is accessed via the model variable directly (which
    IS the ResNet18 instance, not an nn.Sequential wrapper around it).
    """
    target_layer = model.layer4[-1]
    acts:  dict  = {}
    grads: dict  = {}

    h_fwd = target_layer.register_forward_hook(
        lambda m, i, o: acts.update({"v": o.detach()})
    )
    h_bwd = target_layer.register_full_backward_hook(
        lambda m, gi, go: grads.update({"v": go[0].detach()})
    )

    try:
        tensor = _transform(np.ascontiguousarray(face_img)).unsqueeze(0).to(DEVICE)
        model.zero_grad()

        # Temporarily enable gradients for GradCAM pass only
        orig_req_grad = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            logits    = model(tensor)
            fake_logit = logits[0, 1]    # backprop through the fake score
            fake_logit.backward()

        # Restore original requires_grad state
        for p, s in zip(model.parameters(), orig_req_grad):
            p.requires_grad_(s)

        # Global Average Pool gradients → channel weights → weighted activation sum
        weights = grads["v"].mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam     = (weights * acts["v"]).sum(dim=1).squeeze()   # [H, W]
        cam     = torch.relu(cam).cpu().numpy()
        cam     = cv2.resize(cam, (face_img.shape[1], face_img.shape[0]))

        lo, hi  = cam.min(), cam.max()
        cam     = (cam - lo) / (hi - lo + 1e-8)

    except Exception as e:
        print(f"[WARN] GradCAM failed: {e}")
        cam = np.zeros((face_img.shape[0], face_img.shape[1]), dtype=np.float32)

    finally:
        h_fwd.remove()
        h_bwd.remove()

    return cam.astype(np.float32)


# ==========================
# DECISION HELPER
# ==========================

def _label_and_conf(mean_fake_prob: float) -> Tuple[str, float]:
    """Map a mean fake probability in [0,1] → (label, 0-100 confidence)."""
    if mean_fake_prob >= FAKE_THRESHOLD:
        return "Human Fake Face", round(min(mean_fake_prob * 100, 100.0), 2)
    elif mean_fake_prob <= REAL_THRESHOLD:
        return "Human Real Face", round(min((1.0 - mean_fake_prob) * 100, 100.0), 2)
    else:
        conf = round(abs(mean_fake_prob - 0.5) * 200, 2)   # 0–10 % in uncertain band
        return "Uncertain", conf


# ==========================
# MAIN PREDICTION FUNCTION
# ==========================

def predict_image(image_path: str) -> dict:
    """
    Predict whether face(s) in image_path are real or deepfake.

    Returns dict:
        label        — "Human Real Face" | "Human Fake Face" | "Uncertain" | …
        confidence   — float 0-100
        faces        — list of crop file paths
        boxes        — list of (x, y, w, h) tuples
        face_results — list of (label, confidence, heatmap) per face
    """
    print(f"\n[predictor] Processing: {os.path.basename(image_path)}")

    # crop_faces returns (paths, boxes, det_confidences) — 3-tuple
    cropped_paths, face_boxes, det_confs = crop_faces(image_path)

    # ------------------------------------------------------------------
    # NO FACE DETECTED — full-image fallback with confidence cap
    # ------------------------------------------------------------------
    if not cropped_paths:
        print("[predictor] No faces — falling back to full image.")
        img = cv2.imread(image_path)
        if img is None:
            return {"label": "Invalid Image", "confidence": 0.0,
                    "faces": [], "boxes": [], "face_results": []}

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Enhance on full-res BEFORE resize
        if np.std(rgb) < 50:
            rgb = enhance_image(rgb)
        else:
            rgb = gentle_sharpen(rgb)

        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        prob   = predict_with_tta(rgb)
        fake_p = prob[1].item()

        label, conf = _label_and_conf(fake_p)
        # Cap fallback confidence — model was trained on face crops, not full scenes
        conf = min(conf, 70.0)

        return {"label": label, "confidence": round(conf, 2),
                "faces": [], "boxes": [], "face_results": []}

    # ------------------------------------------------------------------
    # FACE(S) DETECTED
    # ------------------------------------------------------------------
    print(f"[predictor] Processing {len(cropped_paths)} face(s).")
    fake_probs:   List[float] = []
    face_weights: List[float] = []
    face_results: list        = []

    for idx, path in enumerate(cropped_paths):
        face = cv2.imread(path) if isinstance(path, str) else path
        if face is None:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.ascontiguousarray(face)

        # Size check BEFORE resize (after resize it is always 224×224)
        if face.shape[0] < MIN_FACE_PX or face.shape[1] < MIN_FACE_PX:
            print(f"  Face {idx}: skipped — too small ({face.shape[1]}×{face.shape[0]})")
            continue

        # Enhance / sharpen on full-res crop BEFORE resize
        if np.std(face) < 50:
            face = enhance_image(face)
        else:
            face = gentle_sharpen(face)

        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        # Blur check on resized crop (consistent resolution)
        blur = laplacian_variance(face_resized)
        if blur < BLUR_THRESHOLD:
            print(f"  Face {idx}: skipped — too blurry (var={blur:.1f})")
            continue

        prob   = predict_with_tta(face_resized)
        real_p = prob[0].item()
        fake_p = prob[1].item()

        print(f"  Face {idx}: real={real_p:.3f}  fake={fake_p:.3f}")

        # Discard if model is completely 50/50 uncertain
        if max(real_p, fake_p) < 0.50:
            continue

        # GradCAM for visualisation
        heatmap = get_gradcam(face_resized)

        # Weight this face by the detector's confidence
        weight = det_confs[idx] if idx < len(det_confs) else 1.0
        fake_probs.append(fake_p)
        face_weights.append(weight)

        if fake_p >= 0.5:
            f_label, f_conf = CLASS_NAMES[1], fake_p * 100
        else:
            f_label, f_conf = CLASS_NAMES[0], real_p * 100

        face_results.append((f_label, round(f_conf, 2), heatmap))

    # ------------------------------------------------------------------
    # AGGREGATE
    # ------------------------------------------------------------------
    if not fake_probs:
        return {"label": "Low Quality / Uncertain", "confidence": 0.0,
                "faces": cropped_paths, "boxes": face_boxes,
                "face_results": []}

    # Weighted mean — high-confidence detections count more
    mean_fake = float(np.average(fake_probs, weights=face_weights))

    print(f"[predictor] Per-face fake probs : {[round(p,3) for p in fake_probs]}")
    print(f"[predictor] Detector weights    : {[round(w,3) for w in face_weights]}")
    print(f"[predictor] Weighted mean fake  : {mean_fake:.3f}")

    final_label, final_conf = _label_and_conf(mean_fake)

    return {
        "label":        final_label,
        "confidence":   final_conf,
        "faces":        cropped_paths,
        "boxes":        face_boxes,
        "face_results": face_results,
    }