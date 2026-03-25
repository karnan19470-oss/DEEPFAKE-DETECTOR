"""
face_cropper.py — Face detection and cropping using OpenCV DNN (Caffe SSD).

Returns a 3-tuple from crop_faces() so predictor.py can use detector
confidence values for weighted voting.

Depends on: utils.py (enhance_image)
"""

import cv2
import numpy as np
import os
import uuid
import urllib.request
from typing import List, Dict, Tuple
from utils import enhance_image

# ==========================
# MODEL PATHS & DOWNLOAD
# ==========================

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
model_path    = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxt_path):
    print("Downloading deploy.prototxt…")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt",
        prototxt_path
    )

if not os.path.exists(model_path):
    print("Downloading face detector model…")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel",
        model_path
    )

face_net   = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
OUTPUT_DIR = os.path.join(BASE_DIR, "cropped_faces")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETECTION_CONFIDENCE = 0.50   # minimum detector score to accept a face
MIN_FACE_PX          = 40     # minimum crop dimension in pixels


# ==========================
# FACE DETECTION
# ==========================

def detect_faces_dnn(rgb_image: np.ndarray) -> List[Dict]:
    """
    Detect faces in an RGB image using the Caffe SSD model.

    Converts RGB → BGR internally for correct blob mean subtraction
    (the Caffe model was trained with BGR means).

    Args:
        rgb_image: full-resolution RGB uint8 array.

    Returns:
        List of dicts, each with:
            'box'        — [x, y, w, h] in pixel coords
            'confidence' — detector confidence (float 0-1)
        Sorted by confidence descending.
    """
    bgr    = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    h, w   = bgr.shape[:2]
    blob   = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    dets = face_net.forward()

    results = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < DETECTION_CONFIDENCE:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
            continue
        results.append({"box": [x1, y1, bw, bh], "confidence": conf})

    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results


# ==========================
# HELPERS
# ==========================

def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


# ==========================
# MAIN FUNCTION
# ==========================

def crop_faces(
    image_path: str,
) -> Tuple[List[str], List[Tuple[int, int, int, int]], List[float]]:
    """
    Detect faces, enhance on full-res, then crop with a context margin.

    Enhancement is applied BEFORE detection on the full-res image so CLAHE
    acts on all available pixel detail (not on a 224×224 downscaled copy).

    Returns:
        face_paths      — list of saved BGR JPEG paths
        valid_boxes     — list of (x, y, w, h) in original image coords
        det_confidences — detector confidence per face (used for weighted voting)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[face_cropper] Cannot read: {image_path}")
        return [], [], []

    # Enhance full-res before detection
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if np.std(rgb) < 50:
        rgb = enhance_image(rgb)

    detections = detect_faces_dnn(rgb)

    if not detections:
        print("[face_cropper] No faces detected.")
        return [], [], []

    h_img, w_img = rgb.shape[:2]
    face_paths, valid_boxes, det_confs = [], [], []
    session_id = uuid.uuid4().hex[:8]   # prevents concurrent-call filename collisions

    for face_id, det in enumerate(detections):
        x, y, w, h = det["box"]

        # Proportional context margins — more forehead helps the model
        mx = int(w * 0.20)
        my_top    = int(h * 0.20)
        my_bottom = int(h * 0.10)

        cx1 = _clamp(x - mx,            0, w_img)
        cx2 = _clamp(x + w + mx,        0, w_img)
        cy1 = _clamp(y - my_top,         0, h_img)
        cy2 = _clamp(y + h + my_bottom,  0, h_img)

        crop = rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        filename = os.path.join(OUTPUT_DIR, f"face_{session_id}_{face_id}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        face_paths.append(filename)
        valid_boxes.append((x, y, w, h))
        det_confs.append(det["confidence"])

    print(f"[face_cropper] {len(face_paths)} face(s) saved.")
    return face_paths, valid_boxes, det_confs