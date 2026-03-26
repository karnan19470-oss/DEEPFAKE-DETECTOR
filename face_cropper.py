"""
face_cropper.py — Enhanced face detection with multiple backends and better cropping
"""

from utils import laplacian_variance
import cv2
import numpy as np
import os
import uuid
import urllib.request
from typing import List, Dict, Tuple, Optional
from utils import enhance_image

# ==========================
# MODEL PATHS & DOWNLOAD
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
model_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Download models if not present
if not os.path.exists(prototxt_path):
    print("Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt",
        prototxt_path
    )

if not os.path.exists(model_path):
    print("Downloading face detector model...")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel",
        model_path
    )

# Initialize face detector
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Alternative: Use OpenCV's built-in face detector as fallback
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

OUTPUT_DIR = os.path.join(BASE_DIR, "cropped_faces")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETECTION_CONFIDENCE = 0.35  # Lower threshold for better recall
MIN_FACE_PX = 40
MAX_FACE_PX = 1000
FACE_PADDING_RATIO = 0.25  # Add padding around faces

# ==========================
# ENHANCED FACE DETECTION
# ==========================

def detect_faces_dnn(rgb_image: np.ndarray, min_confidence: float = DETECTION_CONFIDENCE) -> List[Dict]:
    """
    Detect faces using DNN model with better preprocessing
    """
    h, w = rgb_image.shape[:2]
    
    # Skip if image is too small
    if h < 50 or w < 50:
        return []
    
    # Convert to BGR for DNN
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # Create blob with better preprocessing
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )
    
    face_net.setInput(blob)
    detections = face_net.forward()
    
    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        
        if confidence < min_confidence:
            continue
            
        # Get bounding box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        width = x2 - x1
        height = y2 - y1
        
        # Filter by size
        if width < MIN_FACE_PX or height < MIN_FACE_PX:
            continue
        if width > MAX_FACE_PX or height > MAX_FACE_PX:
            continue
        
        # Check aspect ratio (faces should be roughly square)
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
        
        results.append({
            "box": [x1, y1, width, height],
            "confidence": confidence,
            "center": (x1 + width//2, y1 + height//2)
        })
    
    # Sort by confidence and remove overlapping boxes (NMS)
    results.sort(key=lambda d: d["confidence"], reverse=True)
    
    # Simple NMS to remove overlapping detections
    filtered_results = []
    for result in results:
        x1, y1, w, h = result["box"]
        keep = True
        for existing in filtered_results:
            ex1, ey1, ew, eh = existing["box"]
            # Calculate IoU
            intersection_x1 = max(x1, ex1)
            intersection_y1 = max(y1, ey1)
            intersection_x2 = min(x1 + w, ex1 + ew)
            intersection_y2 = min(y1 + h, ey1 + eh)
            
            if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                union_area = w * h + ew * eh - intersection_area
                iou = intersection_area / union_area if union_area > 0 else 0
                
                if iou > 0.5:  # Overlap threshold
                    keep = False
                    break
        
        if keep:
            filtered_results.append(result)
    
    return filtered_results

def detect_faces_haar(rgb_image: np.ndarray) -> List[Dict]:
    """
    Fallback face detection using Haar cascades
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces with different scales
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    for (x, y, w, h) in faces:
        results.append({
            "box": [x, y, w, h],
            "confidence": 0.6,  # Haar cascade doesn't provide confidence
            "center": (x + w//2, y + h//2)
        })
    
    return results

def detect_faces_combined(rgb_image: np.ndarray) -> List[Dict]:
    """
    Combine multiple detection methods for better accuracy
    """
    # Try DNN first
    faces = detect_faces_dnn(rgb_image)
    
    # If DNN fails, try Haar cascade
    if not faces:
        faces = detect_faces_haar(rgb_image)
    
    # Apply additional filtering
    filtered_faces = []
    for face in faces:
        x, y, w, h = face["box"]
        
        # Check if face is within image bounds
        if x < 0 or y < 0 or x + w > rgb_image.shape[1] or y + h > rgb_image.shape[0]:
            continue
        
        # Check face region for reasonable variation (not just blank)
        face_region = rgb_image[y:y+h, x:x+w]
        if np.std(face_region) < 5:  # Too uniform, likely false positive
            continue
        
        filtered_faces.append(face)
    
    return filtered_faces

# ==========================
# IMPROVED CROPPING
# ==========================

def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))

def crop_face_with_padding(rgb_image: np.ndarray, face_box: List[int], padding_ratio: float = FACE_PADDING_RATIO) -> np.ndarray:
    """
    Crop face with adaptive padding based on image size
    """
    x, y, w, h = face_box
    h_img, w_img = rgb_image.shape[:2]
    
    # Adaptive padding based on face size
    if w < 100:
        padding_ratio = min(padding_ratio * 1.5, 0.4)  # More padding for small faces
    
    # Calculate padding
    pad_x = int(w * padding_ratio)
    pad_y_top = int(h * padding_ratio * 0.8)  # Less padding on top
    pad_y_bottom = int(h * padding_ratio * 1.2)  # More padding on bottom
    
    # Apply padding with bounds checking
    x1 = _clamp(x - pad_x, 0, w_img)
    x2 = _clamp(x + w + pad_x, 0, w_img)
    y1 = _clamp(y - pad_y_top, 0, h_img)
    y2 = _clamp(y + h + pad_y_bottom, 0, h_img)
    
    return rgb_image[y1:y2, x1:x2]

def enhance_face_crop(face_crop: np.ndarray) -> np.ndarray:
    """
    Apply intelligent enhancement to face crops
    """
    # Convert to float for processing
    face_float = face_crop.astype(np.float32) / 255.0
    
    # Apply histogram equalization in LAB space if needed
    if np.std(face_crop) < 40:
        lab = cv2.cvtColor(face_crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        face_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Apply slight sharpening if not too blurry
    blur_score = laplacian_variance(face_crop)
    if blur_score > 15 and blur_score < 40:
        kernel = np.array([[-0.5,-0.5,-0.5],
                          [-0.5, 5,-0.5],
                          [-0.5,-0.5,-0.5]]) / 3.0
        face_crop = cv2.filter2D(face_crop, -1, kernel)
        face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)
    
    return face_crop

# ==========================
# MAIN FUNCTION
# ==========================

def crop_faces(
    image_path: str,
    enhance_crops: bool = True
) -> Tuple[List[str], List[Tuple[int, int, int, int]], List[float]]:
    """
    Enhanced face detection and cropping
    
    Returns:
        face_paths: list of saved crop file paths
        valid_boxes: list of (x, y, w, h) in original image coords
        det_confidences: detector confidence per face
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[face_cropper] Cannot read: {image_path}")
        return [], [], []
    
    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces with multiple methods
    detections = detect_faces_combined(rgb)
    
    if not detections:
        print("[face_cropper] No faces detected")
        return [], [], []
    
    print(f"[face_cropper] Detected {len(detections)} faces")
    
    face_paths = []
    valid_boxes = []
    det_confs = []
    session_id = uuid.uuid4().hex[:8]
    
    for face_id, detection in enumerate(detections):
        box = detection["box"]
        confidence = detection["confidence"]
        
        # Crop with padding
        crop = crop_face_with_padding(rgb, box)
        
        if crop.size == 0:
            continue
        
        # Optional enhancement
        if enhance_crops:
            crop = enhance_face_crop(crop)
        
        # Save crop
        filename = os.path.join(OUTPUT_DIR, f"face_{session_id}_{face_id}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        
        face_paths.append(filename)
        valid_boxes.append(tuple(box))
        det_confs.append(confidence)
        
        print(f"[face_cropper] Saved face {face_id}: size={crop.shape[1]}x{crop.shape[0]}, "
              f"conf={confidence:.3f}")
    
    return face_paths, valid_boxes, det_confs