"""
check_model.py — Diagnostic tool to check model predictions
"""

import os
import cv2
import numpy as np
import torch
from predict_face import predict_image, model, DEVICE, FAKE_THRESHOLD

def check_model_health():
    """Check if model is properly loaded and working"""
    print("=" * 60)
    print("MODEL DIAGNOSTIC CHECK")
    print("=" * 60)
    
    # Check model
    print(f"\n1. Model Status:")
    print(f"   - Model loaded: {model is not None}")
    if model is not None:
        print(f"   - Device: {DEVICE}")
        print(f"   - Model type: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Check if model is in eval mode
        print(f"   - Training mode: {model.training}")
        
    print(f"\n2. Configuration:")
    print(f"   - FAKE_THRESHOLD: {FAKE_THRESHOLD}")
    print(f"   - IMG_SIZE: 224")
    
    print("\n3. Testing with random input...")
    # Test with random input
    random_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(random_input)
        probs = torch.softmax(output, dim=1)
        print(f"   - Output shape: {output.shape}")
        print(f"   - Probabilities: real={probs[0][0]:.3f}, fake={probs[0][1]:.3f}")

def test_with_image(image_path):
    """Test prediction on a specific image"""
    print(f"\n" + "=" * 60)
    print(f"TESTING WITH IMAGE: {os.path.basename(image_path)}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    result = predict_image(image_path)
    
    print(f"\nPrediction Result:")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"  Quality: {result.get('quality', 'N/A')}")
    
    if 'fake_prob' in result:
        print(f"  Fake Probability: {result['fake_prob']:.3f}")
    
    if result.get('face_results'):
        print(f"\n  Per-face results:")
        for i, (label, conf, _) in enumerate(result['face_results']):
            print(f"    Face {i+1}: {label} ({conf:.1f}%)")
    
    return result

if __name__ == "__main__":
    check_model_health()
    
    # Test with a real image if provided
    import sys
    if len(sys.argv) > 1:
        test_with_image(sys.argv[1])
    else:
        print("\nTo test with a specific image, run:")
        print("python check_model.py path/to/image.jpg")