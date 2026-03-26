"""
utils.py — Enhanced image processing utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def laplacian_variance(rgb_image: np.ndarray) -> float:
    """
    Calculate blur score using Laplacian variance
    
    Args:
        rgb_image: RGB image as numpy array
    
    Returns:
        Float variance value (higher = sharper)
    """
    if len(rgb_image.shape) == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_image
    
    # Apply Gaussian blur to reduce noise before Laplacian
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return float(laplacian.var())

def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate noise level in image using wavelet decomposition
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Difference between original and blurred gives noise estimate
    noise = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    
    return float(np.mean(noise))

def enhance_image(image: np.ndarray, method: str = 'auto') -> np.ndarray:
    """
    Enhanced image enhancement with multiple methods
    
    Args:
        image: RGB image as numpy array
        method: 'clahe', 'hist_eq', 'auto', or 'none'
    """
    if method == 'none':
        return image
    
    if method == 'auto':
        # Choose method based on image characteristics
        contrast = np.std(image)
        noise_level = estimate_noise(image)
        
        if contrast < 40:
            method = 'clahe'
        elif noise_level < 15:
            method = 'hist_eq'
        else:
            return gentle_sharpen(image)
    
    if method == 'clahe':
        # Convert to LAB for better color preservation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive CLAHE parameters
        clip_limit = 2.0 if np.std(l) < 50 else 1.5
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    elif method == 'hist_eq':
        # Histogram equalization on each channel
        enhanced = np.zeros_like(image)
        for i in range(3):
            enhanced[:,:,i] = cv2.equalizeHist(image[:,:,i])
        return enhanced
    
    return image

def gentle_sharpen(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Unsharp mask sharpening with controllable strength
    
    Args:
        image: RGB image as numpy array
        strength: Sharpening strength (0-1)
    """
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    
    # Apply unsharp mask
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    
    # Clip to valid range
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def reduce_compression_artifacts(image: np.ndarray) -> np.ndarray:
    """
    Reduce JPEG compression artifacts
    """
    # Apply slight Gaussian blur to reduce blockiness
    blurred = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # Blend original and blurred based on local blockiness
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blockiness = estimate_blockiness(gray)
    
    if blockiness > 0.3:
        alpha = 0.3  # More blur for high blockiness
    elif blockiness > 0.1:
        alpha = 0.1
    else:
        alpha = 0.0
    
    if alpha > 0:
        result = cv2.addWeighted(image, 1 - alpha, blurred, alpha, 0)
        return result.astype(np.uint8)
    
    return image

def estimate_blockiness(gray_image: np.ndarray) -> float:
    """
    Estimate JPEG blockiness artifacts
    """
    h, w = gray_image.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    
    if h8 < 32 or w8 < 32:
        return 0.0
    
    # Extract block boundaries
    block_edges_h = []
    block_edges_v = []
    
    for i in range(8, h8, 8):
        diff = np.abs(gray_image[i-1, :w8].astype(float) - gray_image[i, :w8].astype(float))
        block_edges_h.append(np.mean(diff))
    
    for j in range(8, w8, 8):
        diff = np.abs(gray_image[:h8, j-1].astype(float) - gray_image[:h8, j].astype(float))
        block_edges_v.append(np.mean(diff))
    
    if not block_edges_h or not block_edges_v:
        return 0.0
    
    mean_edge = (np.mean(block_edges_h) + np.mean(block_edges_v)) / 2
    image_std = np.std(gray_image)
    
    if image_std < 1e-6:
        return 0.0
    
    return min(mean_edge / image_std, 1.0)

def detect_facial_landmarks(rgb_image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Simple facial landmark detection using dlib if available
    Returns facial landmarks as numpy array if dlib is available
    """
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        x, y, w, h = face_box
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        landmarks = predictor(gray, dlib_rect)
        
        # Convert to numpy array
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        return points
        
    except (ImportError, Exception):
        return None

def align_face(rgb_image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align face based on eye positions
    """
    if landmarks is None or len(landmarks) < 36:
        return rgb_image
    
    # Get eye coordinates
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    
    # Calculate rotation angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Rotate image
    h, w = rgb_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(rgb_image, rotation_matrix, (w, h))
    
    return aligned

def normalize_face(rgb_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Comprehensive face normalization pipeline
    """
    # Resize
    resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0
    
    # Apply mean subtraction and scaling (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    return normalized