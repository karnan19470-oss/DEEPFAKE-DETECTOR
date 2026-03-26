"""
predict_face.py — Enhanced with confidence calibration for better predictions
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
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_resnet18_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
FAKE_THRESHOLD = 0.5  # Balanced threshold

# Quality thresholds
MIN_FACE_PX = 40
BLUR_THRESHOLD = 20
MIN_CONTRAST = 15
MAX_CONTRAST = 200

# ==========================
# MODEL ARCHITECTURE
# ==========================

def build_model():
    """Build model architecture matching pre-trained model"""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    return model

def load_model():
    """Load your pre-trained model correctly"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Creating new untrained model...")
        return build_model()
    
    print(f"✅ Found model at: {MODEL_PATH}")
    
    try:
        model = build_model()
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            print(f"📦 Loading state dict with {len(state_dict)} keys")
            model.load_state_dict(state_dict, strict=True)
            print("✅ Model loaded successfully!")
            
            model.eval()
            
            # Test model output
            test_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            with torch.no_grad():
                test_output = model(test_input)
                test_probs = F.softmax(test_output, dim=1)
                print(f"🧪 Test inference - Real: {test_probs[0][0]:.3f}, Fake: {test_probs[0][1]:.3f}")
            
            return model
        else:
            raise ValueError("Invalid checkpoint format")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return build_model()

# Load model
model = load_model()
model.to(DEVICE)
model.eval()
print(f"📱 Model on: {DEVICE}")

# ==========================
# TRANSFORMS
# ==========================

inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================
# UTILITY FUNCTIONS
# ==========================

def laplacian_variance_safe(rgb_image):
    """Safe wrapper for laplacian_variance"""
    try:
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except:
        return 25.0

def run_inference(img_tensor):
    """Run inference"""
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]
    return probs[0].item(), probs[1].item()

def predict_with_calibration(face_img, use_tta=True):
    """
    Enhanced prediction with calibration and optional TTA
    """
    try:
        if isinstance(face_img, np.ndarray):
            if face_img.dtype != np.uint8:
                face_img = (face_img * 255).astype(np.uint8)
            
            if use_tta:
                # Test Time Augmentation
                predictions = []
                
                # Original
                tensor = inference_transform(face_img).unsqueeze(0).to(DEVICE)
                real_p, fake_p = run_inference(tensor)
                predictions.append(fake_p)
                
                # Horizontal flip
                flipped = cv2.flip(face_img, 1)
                tensor = inference_transform(flipped).unsqueeze(0).to(DEVICE)
                real_p, fake_p = run_inference(tensor)
                predictions.append(fake_p)
                
                # Slight rotation (if image is large enough)
                if min(face_img.shape[:2]) > 100:
                    h, w = face_img.shape[:2]
                    center = (w//2, h//2)
                    matrix = cv2.getRotationMatrix2D(center, 3, 1.0)
                    rotated = cv2.warpAffine(face_img, matrix, (w, h))
                    tensor = inference_transform(rotated).unsqueeze(0).to(DEVICE)
                    real_p, fake_p = run_inference(tensor)
                    predictions.append(fake_p)
                
                # Average predictions
                fake_prob = np.mean(predictions)
                
                # Quality-based adjustment
                blur = laplacian_variance_safe(face_img)
                contrast = np.std(face_img)
                
                # Lower confidence for poor quality images
                if blur < 30 or contrast < 30:
                    fake_prob = fake_prob * 0.8 + 0.5 * 0.2  # Pull toward 0.5
                
                real_prob = 1 - fake_prob
                
            else:
                tensor = inference_transform(face_img).unsqueeze(0).to(DEVICE)
                real_prob, fake_prob = run_inference(tensor)
            
            # Apply sigmoid calibration for better probabilities
            calibrated_fake = 1 / (1 + np.exp(-5 * (fake_prob - 0.5)))
            
            return real_prob, calibrated_fake
        else:
            return 0.5, 0.5
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0.5, 0.5

# Aliases for compatibility
predict_with_ensemble = predict_with_calibration
predict_single = lambda x: predict_with_calibration(x, use_tta=False)

def assess_quality(rgb_img):
    """Comprehensive quality assessment"""
    try:
        h, w = rgb_img.shape[:2]
        if h < MIN_FACE_PX or w < MIN_FACE_PX:
            return False, f"Too small ({w}x{h})"
        
        blur = laplacian_variance_safe(rgb_img)
        if blur < BLUR_THRESHOLD:
            return False, f"Too blurry (var={blur:.1f})"
        
        contrast = np.std(rgb_img)
        if contrast < MIN_CONTRAST:
            return False, f"Too low contrast ({contrast:.1f})"
        if contrast > MAX_CONTRAST:
            return False, f"Too high contrast ({contrast:.1f})"
        
        return True, f"Good quality (blur={blur:.1f}, contrast={contrast:.1f})"
        
    except Exception as e:
        return False, f"Quality check error: {e}"

def detect_blockiness(gray_img):
    """Detect JPEG blockiness artifacts"""
    try:
        h, w = gray_img.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        
        if h8 < 64 or w8 < 64:
            return 0.0
        
        block_edges = []
        for i in range(8, h8, 8):
            diff = np.abs(gray_img[i-1:i+1, :w8].astype(float))
            if len(diff) > 1:
                block_edges.append(np.mean(np.abs(np.diff(diff, axis=0))))
        
        if not block_edges:
            return 0.0
        
        blockiness = np.mean(block_edges) / (np.std(gray_img) + 1e-6)
        return min(blockiness, 1.0)
        
    except Exception as e:
        return 0.0

# ==========================
# MAIN PREDICTION FUNCTION
# ==========================

def predict_image(image_path):
    """
    Main prediction with enhanced accuracy
    """
    try:
        # Read and validate image
        img = cv2.imread(image_path)
        if img is None:
            return {
                "label": "Invalid Image",
                "confidence": 0,
                "faces": [],
                "boxes": [],
                "face_results": [],
                "quality": "Cannot read image file",
                "error": True
            }
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        try:
            faces, boxes, det_confs = crop_faces(image_path)
        except Exception as e:
            print(f"Face detection error: {e}")
            faces, boxes, det_confs = [], [], []
        
        # No faces detected - fallback to full image
        if len(faces) == 0:
            print("No faces detected - analyzing full image")
            
            # Quality check
            is_good, quality_msg = assess_quality(rgb)
            
            # Apply enhancement
            try:
                if np.std(rgb) < 50:
                    from utils import enhance_image
                    rgb = enhance_image(rgb)
            except:
                pass
            
            # Prediction with TTA
            real_p, fake_p = predict_with_calibration(rgb, use_tta=True)
            
            # Determine final verdict
            if fake_p >= FAKE_THRESHOLD:
                label = "Human Fake Face"
                conf = fake_p * 100
            else:
                label = "Human Real Face"
                conf = (1 - fake_p) * 100
            
            return {
                "label": label,
                "confidence": round(conf, 2),
                "faces": [],
                "boxes": [],
                "face_results": [],
                "quality": quality_msg,
                "fake_prob": fake_p,
                "error": False
            }
        
        print(f"\n🔍 Found {len(faces)} face(s)")
        
        # Process each face with enhanced voting
        face_predictions = []
        
        for idx, face_path in enumerate(faces):
            try:
                face_img = cv2.imread(face_path)
                if face_img is None:
                    continue
                
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Quality check
                is_good, quality_msg = assess_quality(face_rgb)
                if not is_good:
                    print(f"  Face {idx}: {quality_msg} - skipping")
                    continue
                
                # Enhance if needed
                try:
                    if np.std(face_rgb) < 50:
                        from utils import enhance_image
                        face_rgb = enhance_image(face_rgb)
                except:
                    pass
                
                # Prediction with TTA
                real_p, fake_p = predict_with_calibration(face_rgb, use_tta=True)
                
                # Get detection confidence
                det_conf = det_confs[idx] if idx < len(det_confs) else 0.7
                
                # Area weight (larger faces more important)
                if idx < len(boxes):
                    area = boxes[idx][2] * boxes[idx][3]
                    total_area = sum(b[2] * b[3] for b in boxes) if boxes else 1
                    area_weight = area / total_area if total_area > 0 else 0.5
                else:
                    area_weight = 0.5
                
                # Combined weight
                weight = det_conf * (0.6 + 0.4 * area_weight)
                
                face_predictions.append({
                    'fake_prob': fake_p,
                    'weight': weight,
                    'real_prob': real_p,
                    'area_weight': area_weight
                })
                
                # Per-face result
                if fake_p >= FAKE_THRESHOLD:
                    f_label = "Fake"
                    f_conf = fake_p * 100
                    status = "🚨"
                else:
                    f_label = "Real"
                    f_conf = (1 - fake_p) * 100
                    status = "✅"
                
                print(f"  Face {idx}: {status} {f_label} ({f_conf:.1f}%) | "
                      f"fake={fake_p:.3f} | weight={weight:.2f}")
                
            except Exception as e:
                print(f"  Face {idx}: Error - {e}")
                continue
        
        # No usable faces
        if len(face_predictions) == 0:
            return {
                "label": "No Usable Faces",
                "confidence": 0,
                "faces": faces,
                "boxes": boxes,
                "face_results": [],
                "quality": "No faces passed quality checks",
                "error": True
            }
        
        # Weighted voting
        total_weight = sum(p['weight'] for p in face_predictions)
        if total_weight > 0:
            weighted_fake = sum(p['fake_prob'] * p['weight'] for p in face_predictions) / total_weight
        else:
            weighted_fake = 0.5
        
        # Calculate confidence based on agreement
        fake_probs = [p['fake_prob'] for p in face_predictions]
        agreement = 1 - np.std(fake_probs) if len(fake_probs) > 1 else 1
        
        # Final probability with agreement boost
        if agreement > 0.8:
            # High agreement - boost confidence
            final_fake = weighted_fake
            confidence_boost = 1.2
        elif agreement < 0.5:
            # Low agreement - reduce confidence
            final_fake = weighted_fake * 0.7 + 0.5 * 0.3
            confidence_boost = 0.8
        else:
            final_fake = weighted_fake
            confidence_boost = 1.0
        
        print(f"\n📊 Analysis Results:")
        print(f"  Weighted fake probability: {weighted_fake:.3f}")
        print(f"  Face agreement: {agreement:.3f}")
        print(f"  Final probability: {final_fake:.3f}")
        
        # Final verdict
        if final_fake >= FAKE_THRESHOLD:
            final_label = "Human Fake Face"
            final_conf = min(final_fake * 100 * confidence_boost, 99.9)
            print(f"🚨 VERDICT: FAKE with {final_conf:.1f}% confidence")
        else:
            final_label = "Human Real Face"
            final_conf = min((1 - final_fake) * 100 * confidence_boost, 99.9)
            print(f"✅ VERDICT: REAL with {final_conf:.1f}% confidence")
        
        # Prepare face results
        face_results = []
        for i, pred in enumerate(face_predictions):
            if pred['fake_prob'] >= FAKE_THRESHOLD:
                f_label = "Human Fake Face"
                f_conf = pred['fake_prob'] * 100
            else:
                f_label = "Human Real Face"
                f_conf = (1 - pred['fake_prob']) * 100
            face_results.append((f_label, round(f_conf, 2), None))
        
        return {
            "label": final_label,
            "confidence": round(final_conf, 2),
            "faces": faces,
            "boxes": boxes,
            "face_results": face_results,
            "quality": f"Analyzed {len(face_predictions)} faces (agreement={agreement:.2f})",
            "fake_prob": final_fake,
            "error": False
        }
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "label": "Error",
            "confidence": 0,
            "faces": [],
            "boxes": [],
            "face_results": [],
            "quality": f"Error: {str(e)}",
            "error": True
        }

print("\n" + "="*60)
print("✅ PREDICTION ENGINE READY")
print(f"🎯 Fake threshold: {FAKE_THRESHOLD}")
print(f"🔄 TTA enabled: Yes")
print("="*60 + "\n")