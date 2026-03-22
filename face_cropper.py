import cv2
import numpy as np
import os
import urllib.request

# File paths
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

# Download if not exists
if not os.path.exists(prototxt_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        prototxt_path
    )

if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        model_path
    )

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

OUTPUT_DIR = "cropped_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONTEXT_RATIO = 0.15

# ==========================
# IMAGE ENHANCEMENT
# ==========================

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return enhanced

# ==========================
# BLUR CHECK
# ==========================

def is_blurry(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold

# ==========================
# MULTI-SCALE DETECTION
# ==========================
def detect_faces_dnn(image):
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            results.append({
                "box": [x1, y1, x2 - x1, y2 - y1],
                "confidence": confidence
            })

    return results
# ==========================
# SAFE BOX CLAMP
# ==========================

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# ==========================
# MAIN FUNCTION
# ==========================

def crop_faces(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Error: Cannot read image")
        return [], []

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = enhance_image(rgb)

    detections = detector.detect_faces(rgb)

    if len(detections) == 0:
        print("⚠️ No faces detected")
        return [], []

    face_paths = []
    valid_boxes = []

    h_img, w_img, _ = rgb.shape
    face_id = 0

    for det in detections:

        x, y, w, h = det['box']
        confidence = det['confidence']

        if confidence < 0.7:
            continue

        if w < 50 or h < 50:
            continue

        x1 = clamp(x, 0, w_img)
        y1 = clamp(y, 0, h_img)
        x2 = clamp(x + w, 0, w_img)
        y2 = clamp(y + h, 0, h_img)

        face_crop = rgb[y1:y2, x1:x2]

        top_margin = int(h * 0.15)
        bottom_margin = int(h * 0.1)
        side_margin = int(w * 0.1)

        cx1 = clamp(x - side_margin, 0, w_img)
        cx2 = clamp(x + w + side_margin, 0, w_img)
        cy1 = clamp(y - top_margin, 0, h_img)
        cy2 = clamp(y + h + bottom_margin, 0, h_img)

        context_crop = rgb[cy1:cy2, cx1:cx2]
        if context_crop is None or context_crop.size == 0:
            continue
    

        context_crop = cv2.cvtColor(context_crop, cv2.COLOR_RGB2BGR)

        filename = os.path.join(
            OUTPUT_DIR,
            f"face_{face_id}.jpg"
        )

        cv2.imwrite(filename, context_crop)

        face_paths.append(context_crop)
        valid_boxes.append((x, y, w, h))

        face_id += 1

    print(f"✅ {len(face_paths)} face(s) cropped with context")

    return face_paths, valid_boxes