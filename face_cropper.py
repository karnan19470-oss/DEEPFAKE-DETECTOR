import cv2
import numpy as np
import os
import uuid
import urllib.request
from typing import List, Dict, Tuple
from utils import enhance_image

# ==========================
# PATHS — resolve relative to this file so download location == load location
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# FIX 1: Download to BASE_DIR paths from the start so readNetFromCaffe
# reads the same file that was downloaded (previously CWD != BASE_DIR caused crash).
if not os.path.exists(prototxt_path):
    print("Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        prototxt_path
    )

if not os.path.exists(model_path):
    print("Downloading face detector model...")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        model_path
    )

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

OUTPUT_DIR = os.path.join(BASE_DIR, "cropped_faces")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FIX 9: Use typing.List / Dict for Python 3.8 compatibility (list[dict] requires 3.9+).
DETECTION_CONFIDENCE = 0.5
MIN_FACE_PX = 30


# ==========================
# DNN FACE DETECTION
# ==========================

def detect_faces_dnn(image: np.ndarray) -> List[Dict]:
    """
    Detect faces using the Caffe SSD model.

    Args:
        image: RGB uint8 image.

    Returns:
        List of dicts with keys 'box' ([x, y, w, h]) and 'confidence' (float).
        Only detections above DETECTION_CONFIDENCE with size >= MIN_FACE_PX are returned.
    """
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < DETECTION_CONFIDENCE:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
            continue
        results.append({"box": [x1, y1, bw, bh], "confidence": conf})

    return results


# ==========================
# HELPERS
# ==========================

def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


# ==========================
# MAIN FUNCTION
# ==========================

def crop_faces(image_path: str) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """
    Detect and crop faces from image_path with a small context margin.

    Returns:
        face_paths  — list of saved crop file paths
        valid_boxes — list of (x, y, w, h) tuples in original image coords
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image at {image_path}")
        return [], []

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = enhance_image(rgb)

    detections = detect_faces_dnn(rgb)

    if not detections:
        print("No faces detected — caller should handle full-image fallback")
        return [], []

    h_img, w_img = rgb.shape[:2]
    face_paths, valid_boxes = [], []

    # FIX 5: Use a unique session prefix per crop_faces call so concurrent
    # calls (video mode, multi-user) don't overwrite each other's face_0.jpg.
    session_id = uuid.uuid4().hex[:8]

    for face_id, det in enumerate(detections):
        x, y, w, h = det["box"]

        # Context margins
        top_margin    = int(h * 0.15)
        bottom_margin = int(h * 0.10)
        side_margin   = int(w * 0.10)

        cx1 = clamp(x - side_margin,     0, w_img)
        cx2 = clamp(x + w + side_margin,  0, w_img)
        cy1 = clamp(y - top_margin,       0, h_img)
        cy2 = clamp(y + h + bottom_margin, 0, h_img)

        crop = rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:  # FIX 6: numpy slice is never None, only check .size
            continue

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        filename = os.path.join(OUTPUT_DIR, f"face_{session_id}_{face_id}.jpg")
        cv2.imwrite(filename, crop_bgr)

        face_paths.append(filename)
        valid_boxes.append((x, y, w, h))

    print(f"{len(face_paths)} face(s) cropped")
    return face_paths, valid_boxes