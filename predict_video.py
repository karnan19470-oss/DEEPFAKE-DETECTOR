"""
predict_video.py — Fixed video prediction for better real video accuracy
"""

import os
import shutil
import numpy as np
import cv2
from typing import Callable, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from face_cropper import detect_faces_combined
from utils import enhance_image, gentle_sharpen, laplacian_variance, reduce_compression_artifacts

# Import from predict_face
from predict_face import (
    predict_with_calibration,
    assess_quality,
    IMG_SIZE,
    FAKE_THRESHOLD,
    inference_transform
)

# ==========================
# CONFIGURATION - ADJUSTED FOR BETTER REAL VIDEO ACCURACY
# ==========================

SAMPLE_EVERY_N_SECONDS = 0.5
MAX_FRAMES = 300
MIN_FRAMES_FOR_VERDICT = 3
MIN_FACE_PX = 50
BLUR_THRESHOLD = 18  # Lowered to accept slightly blurry real videos
MIN_FRAME_CONFIDENCE = 0.40  # Lowered threshold
OUTLIER_STD_MULT = 2.0
TEMPORAL_SMOOTHING_WINDOW = 7  # Increased for smoother transitions

# NEW: Bias correction for real videos
REAL_VIDEO_BIAS = -0.08  # Slight bias toward real (negative = more likely real)
CONFIDENCE_THRESHOLD_HIGH = 0.75  # High confidence threshold
CONFIDENCE_THRESHOLD_LOW = 0.45   # Low confidence threshold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# TEMPORAL ANALYSIS
# ==========================

class TemporalAnalyzer:
    """Analyze temporal consistency of predictions"""
    
    def __init__(self, window_size: int = TEMPORAL_SMOOTHING_WINDOW):
        self.window_size = window_size
        self.history = []
    
    def add_prediction(self, fake_prob: float, confidence: float, frame_quality: float):
        """Add new prediction to history with confidence and quality"""
        self.history.append({
            'fake_prob': fake_prob,
            'confidence': confidence,
            'quality': frame_quality,
            'timestamp': len(self.history)
        })
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_smoothed(self) -> Dict:
        """Get temporally smoothed prediction with confidence"""
        if not self.history:
            return {'fake_prob': 0.5, 'confidence': 0.0, 'consistency': 0.0}
        
        # Weighted average (more weight to recent frames and high quality frames)
        weights = []
        for i, h in enumerate(self.history):
            # Recent frames get more weight
            recency_weight = np.exp(i / len(self.history))
            # High quality frames get more weight
            quality_weight = h['quality'] + 0.5
            weights.append(recency_weight * quality_weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_fake = sum(h['fake_prob'] * w for h, w in zip(self.history, weights))
        
        # Calculate temporal consistency
        fake_probs = [h['fake_prob'] for h in self.history]
        temporal_std = np.std(fake_probs) if len(fake_probs) > 1 else 0
        
        # Higher confidence if predictions are consistent
        consistency_score = 1 - min(temporal_std, 0.5)
        avg_confidence = np.mean([h['confidence'] for h in self.history])
        
        combined_confidence = avg_confidence * (0.7 + 0.3 * consistency_score)
        
        return {
            'fake_prob': weighted_fake,
            'confidence': combined_confidence,
            'consistency': consistency_score,
            'temporal_std': temporal_std,
            'trend': self._get_trend()
        }
    
    def _get_trend(self) -> str:
        """Determine trend of predictions"""
        if len(self.history) < 5:
            return 'stable'
        
        recent = [h['fake_prob'] for h in self.history[-3:]]
        older = [h['fake_prob'] for h in self.history[:3]]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg + 0.1:
            return 'increasing_fake'
        elif recent_avg < older_avg - 0.1:
            return 'decreasing_fake'
        else:
            return 'stable'
    
    def is_consistent(self) -> bool:
        """Check if predictions are consistent"""
        if len(self.history) < 3:
            return True
        fake_probs = [h['fake_prob'] for h in self.history]
        return np.std(fake_probs) < 0.2  # Increased tolerance

# ==========================
# ENHANCED FRAME ANALYSIS
# ==========================

def analyze_frame_enhanced(rgb_frame: np.ndarray, use_tta: bool = True) -> Optional[Dict]:
    """
    Enhanced frame analysis with quality-aware prediction
    """
    # Detect faces
    detections = detect_faces_combined(rgb_frame)
    
    if not detections:
        return None
    
    # Collect all faces for voting
    face_predictions = []
    h_img, w_img = rgb_frame.shape[:2]
    
    for detection in detections:
        x, y, w, h = detection["box"]
        conf = detection["confidence"]
        
        # Skip very small faces
        if w < MIN_FACE_PX or h < MIN_FACE_PX:
            continue
        
        # Extract face region with padding
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        
        x1 = max(0, x - pad_x)
        x2 = min(w_img, x + w + pad_x)
        y1 = max(0, y - pad_y)
        y2 = min(h_img, y + h + pad_y)
        
        face_region = rgb_frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            continue
        
        # Quality assessment
        blur_score = laplacian_variance(face_region)
        contrast = np.std(face_region)
        
        # Skip very blurry or low contrast faces
        if blur_score < BLUR_THRESHOLD or contrast < 10:
            continue
        
        # Apply minimal enhancement for real videos (avoid over-processing)
        try:
            if contrast < 30:
                face_region = enhance_image(face_region, method='clahe')
            elif blur_score < 30:
                face_region = gentle_sharpen(face_region, strength=0.2)
        except:
            pass
        
        # Resize for model
        face_resized = cv2.resize(face_region, (IMG_SIZE, IMG_SIZE))
        
        # Run prediction
        try:
            real_p, fake_p = predict_with_calibration(face_resized, use_tta=use_tta)
            
            # Apply real video bias (makes it more likely to be real)
            fake_p = max(0, min(1, fake_p + REAL_VIDEO_BIAS))
            
            # Calculate quality score
            quality_score = min(blur_score / 40.0, 1.0) * min(contrast / 50.0, 1.0)
            
            # Area weight (larger faces more important)
            area = w * h
            area_weight = area / (max(d["box"][2] * d["box"][3] for d in detections) if detections else 1)
            
            # Combined weight
            weight = conf * (0.5 + 0.5 * area_weight) * quality_score
            
            face_predictions.append({
                'fake_prob': fake_p,
                'weight': weight,
                'quality': quality_score,
                'area': area
            })
        except Exception as e:
            continue
    
    if not face_predictions:
        return None
    
    # Weighted voting
    total_weight = sum(p['weight'] for p in face_predictions)
    if total_weight > 0:
        weighted_fake = sum(p['fake_prob'] * p['weight'] for p in face_predictions) / total_weight
    else:
        weighted_fake = 0.5
    
    # Calculate confidence from face agreement
    fake_probs = [p['fake_prob'] for p in face_predictions]
    agreement = 1 - np.std(fake_probs) if len(fake_probs) > 1 else 1
    
    # Calculate frame quality
    avg_quality = np.mean([p['quality'] for p in face_predictions])
    
    return {
        "fake_prob": weighted_fake,
        "confidence": agreement * avg_quality,
        "num_faces": len(face_predictions),
        "agreement": agreement,
        "frame_quality": avg_quality,
        "face_std": np.std(fake_probs) if len(fake_probs) > 1 else 0
    }

# ==========================
# MAIN VIDEO PREDICTION
# ==========================

def predict_video(
    video_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict:
    """
    Enhanced video prediction with bias correction for real videos
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "label": "Error",
            "confidence": 0.0,
            "frames_used": 0,
            "fake_prob": 0.0,
            "frame_log": [],
            "error": "Cannot open video file"
        }
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"🎬 VIDEO ANALYSIS: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    print(f"  FPS: {fps:.1f}, Duration: {duration:.1f}s")
    print(f"  FAKE_THRESHOLD: {FAKE_THRESHOLD}")
    print(f"  REAL_VIDEO_BIAS: {REAL_VIDEO_BIAS}")
    
    # Calculate sampling interval
    frame_interval = max(1, int(round(fps * SAMPLE_EVERY_N_SECONDS)))
    max_samples = min(MAX_FRAMES, total_frames // frame_interval)
    
    frame_id = 0
    sampled = 0
    frame_results = []
    temporal_analyzer = TemporalAnalyzer()
    
    while cap.isOpened() and sampled < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Sample at intervals
        if frame_id % frame_interval != 0:
            continue
        
        sampled += 1
        
        # Update progress
        if progress_callback:
            progress = min(frame_id / total_frames, 0.99)
            progress_callback(progress, f"Frame {frame_id}/{total_frames}")
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze frame
        result = analyze_frame_enhanced(rgb_frame, use_tta=True)
        
        if result:
            fake_prob = result["fake_prob"]
            frame_confidence = result["confidence"]
            frame_quality = result["frame_quality"]
            
            # Add to temporal analyzer
            temporal_analyzer.add_prediction(fake_prob, frame_confidence, frame_quality)
            smoothed = temporal_analyzer.get_smoothed()
            
            frame_results.append({
                "frame_idx": frame_id,
                "timestamp": frame_id / fps,
                "fake_prob_raw": fake_prob,
                "fake_prob_smoothed": smoothed['fake_prob'],
                "confidence": frame_confidence,
                "num_faces": result['num_faces'],
                "agreement": result['agreement'],
                "quality": frame_quality
            })
            
            # Determine per-frame verdict
            frame_verdict = "FAKE" if fake_prob >= FAKE_THRESHOLD else "REAL"
            frame_conf = abs(fake_prob - 0.5) * 200
            
            print(f"  Frame {frame_id:5d}: {frame_verdict} ({frame_conf:.1f}%) | "
                  f"fake={fake_prob:.3f} | faces={result['num_faces']} | "
                  f"agree={result['agreement']:.2f}")
    
    cap.release()
    
    # Clean up crops
    crops_dir = os.path.join(BASE_DIR, "cropped_faces")
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir)
    os.makedirs(crops_dir, exist_ok=True)
    
    # Check if we have enough frames
    if len(frame_results) < MIN_FRAMES_FOR_VERDICT:
        return {
            "label": "Insufficient Data",
            "confidence": 0.0,
            "frames_used": len(frame_results),
            "fake_prob": 0.0,
            "frame_log": [],
            "error": f"Only {len(frame_results)} usable frames"
        }
    
    # Extract smoothed probabilities
    smoothed_probs = [r["fake_prob_smoothed"] for r in frame_results]
    
    # Remove outliers
    if len(smoothed_probs) >= 6:
        arr = np.array(smoothed_probs)
        mean = arr.mean()
        std = arr.std()
        if std > 0.05:
            filtered = [p for p in smoothed_probs if abs(p - mean) <= OUTLIER_STD_MULT * std]
            if len(filtered) >= MIN_FRAMES_FOR_VERDICT:
                smoothed_probs = filtered
                print(f"\n  Removed {len(frame_results) - len(smoothed_probs)} outlier frames")
    
    # Calculate statistics
    probs_array = np.array(smoothed_probs)
    mean_prob = float(probs_array.mean())
    median_prob = float(np.median(probs_array))
    std_prob = float(probs_array.std())
    
    # Get temporal statistics
    temporal_stats = temporal_analyzer.get_smoothed()
    is_consistent = temporal_analyzer.is_consistent()
    trend = temporal_stats['trend']
    
    # Smart combination based on consistency and trend
    if is_consistent:
        if trend == 'decreasing_fake':
            # If fake probability is decreasing, bias toward real
            combined_prob = mean_prob * 0.7 + median_prob * 0.3
            combined_prob = combined_prob * 0.9  # Reduce fake probability
        elif trend == 'increasing_fake':
            combined_prob = mean_prob * 0.6 + median_prob * 0.4
        else:
            combined_prob = mean_prob * 0.6 + median_prob * 0.4
    else:
        # Inconsistent - trust median more
        combined_prob = mean_prob * 0.4 + median_prob * 0.6
    
    # Apply variance penalty
    variance_penalty = min(std_prob * 0.5, 0.15)
    combined_prob = combined_prob * (1 - variance_penalty)
    
    # Final bias for real videos
    if combined_prob < 0.6:  # If not strongly fake
        combined_prob = combined_prob * 0.95  # Slight bias toward real
    
    print(f"\n{'='*60}")
    print(f"📊 VIDEO ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Frames analyzed: {len(smoothed_probs)}")
    print(f"  Mean probability: {mean_prob:.3f}")
    print(f"  Median probability: {median_prob:.3f}")
    print(f"  Std deviation: {std_prob:.3f}")
    print(f"  Temporal consistency: {temporal_stats['consistency']:.3f}")
    print(f"  Trend: {trend}")
    print(f"  Combined probability: {combined_prob:.3f}")
    print(f"  Threshold: {FAKE_THRESHOLD}")
    
    # Determine verdict with confidence
    if combined_prob >= FAKE_THRESHOLD:
        # Fake video
        label = "Fake Video"
        confidence = min(combined_prob * 100, 99.9)
        
        # Adjust confidence based on consistency
        if not is_consistent:
            confidence = confidence * 0.85
        if trend == 'decreasing_fake':
            confidence = confidence * 0.9
        
        print(f"\n🚨 VERDICT: {label} with {confidence:.1f}% confidence")
    else:
        # Real video - boost confidence
        label = "Real Video"
        real_prob = 1 - combined_prob
        confidence = min(real_prob * 100, 99.9)
        
        # Boost confidence for consistent real predictions
        if is_consistent and combined_prob < 0.45:
            confidence = min(confidence * 1.15, 99.9)
        if trend == 'decreasing_fake':
            confidence = min(confidence * 1.1, 99.9)
        
        print(f"\n✅ VERDICT: {label} with {confidence:.1f}% confidence")
    
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "frames_used": len(smoothed_probs),
        "fake_prob": round(combined_prob, 4),
        "frame_log": frame_results,
        "error": None,
        "statistics": {
            "mean": mean_prob,
            "median": median_prob,
            "std": std_prob,
            "temporal_consistency": temporal_stats['consistency'],
            "trend": trend,
            "frames_analyzed": len(frame_results)
        }
    }