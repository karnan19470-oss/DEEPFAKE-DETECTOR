"""
diagnose_model.py — Check if model is working properly
"""

import os
import torch
import cv2
import numpy as np
from predict_face import model, DEVICE, FAKE_THRESHOLD, IMG_SIZE, inference_transform

def test_model_weights():
    """Check if model weights are random or trained"""
    print("=" * 60)
    print("MODEL WEIGHT ANALYSIS")
    print("=" * 60)
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Check parameter distribution
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy().flatten()
            all_weights.extend(weights)
            print(f"\n{name}:")
            print(f"  Mean: {weights.mean():.6f}")
            print(f"  Std: {weights.std():.6f}")
            print(f"  Min: {weights.min():.6f}")
            print(f"  Max: {weights.max():.6f}")
    
    all_weights = np.array(all_weights)
    print(f"\nOverall Statistics:")
    print(f"  Mean: {all_weights.mean():.6f}")
    print(f"  Std: {all_weights.std():.6f}")
    
    # Check if weights look like random initialization
    if abs(all_weights.mean()) < 0.01 and all_weights.std() < 0.1:
        print("\n⚠️ WARNING: Weights look like random initialization!")
        print("   The model may not be trained properly.")
    else:
        print("\n✅ Weights show variation - model appears to be trained.")

def test_model_outputs():
    """Test model outputs with random inputs"""
    print("\n" + "=" * 60)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 60)
    
    # Test with random inputs
    print("\nTesting with random inputs:")
    for i in range(5):
        random_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        with torch.no_grad():
            output = model(random_input)
            probs = torch.softmax(output, dim=1)
            print(f"  Test {i+1}: Real={probs[0][0]:.3f}, Fake={probs[0][1]:.3f}")
    
    # Test with same input multiple times
    print("\nTesting consistency with same input:")
    test_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    outputs = []
    for i in range(5):
        with torch.no_grad():
            output = model(test_input)
            probs = torch.softmax(output, dim=1)
            outputs.append(probs[0][1].item())
            print(f"  Run {i+1}: Fake={probs[0][1]:.3f}")
    
    variance = np.var(outputs)
    print(f"\nVariance across runs: {variance:.6f}")
    if variance < 0.0001:
        print("✅ Model outputs are consistent")
    else:
        print("⚠️ Model outputs vary - possible dropout during inference?")

def test_with_real_image(image_path):
    """Test model with actual image"""
    print("\n" + "=" * 60)
    print(f"TESTING WITH ACTUAL IMAGE: {os.path.basename(image_path)}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Test with different preprocessing
    print("\nTesting different preprocessing approaches:")
    
    # 1. Raw image
    tensor = inference_transform(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        print(f"  Raw image: Real={probs[0][0]:.3f}, Fake={probs[0][1]:.3f}")
    
    # 2. Normalized image (0-1 range)
    normalized = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        print(f"  Normalized [-1,1]: Real={probs[0][0]:.3f}, Fake={probs[0][1]:.3f}")
    
    # 3. Different sizes
    for size in [112, 224, 448]:
        resized = cv2.resize(rgb, (size, size))
        tensor = inference_transform(resized).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            print(f"  Size {size}x{size}: Real={probs[0][0]:.3f}, Fake={probs[0][1]:.3f}")
    
    # 4. Check if output changes with different inputs
    print("\nTesting if model distinguishes different inputs:")
    test_inputs = []
    for i in range(3):
        if i == 0:
            test_input = rgb
        elif i == 1:
            # Brightness increased
            test_input = np.clip(rgb * 1.5, 0, 255).astype(np.uint8)
        else:
            # Blurred
            test_input = cv2.GaussianBlur(rgb, (15, 15), 0)
        
        tensor = inference_transform(test_input).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            print(f"  Variation {i+1}: Real={probs[0][0]:.3f}, Fake={probs[0][1]:.3f}")

def check_model_file():
    """Check if model file exists and is valid"""
    print("\n" + "=" * 60)
    print("MODEL FILE CHECK")
    print("=" * 60)
    
    model_path = os.path.join(os.path.dirname(__file__), "deepfake_resnet18_best.pth")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path}")
        print(f"   File size: {file_size:.2f} MB")
        
        # Try to load and inspect
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"   Checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                print(f"   Keys in checkpoint: {list(checkpoint.keys())[:10]}")
                
                # Check for state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                if isinstance(state_dict, dict):
                    num_layers = len(state_dict)
                    print(f"   Number of layers: {num_layers}")
                    
                    # Check first few layer names
                    first_keys = list(state_dict.keys())[:5]
                    print(f"   First layers: {first_keys}")
            else:
                print(f"   Unexpected checkpoint format")
                
        except Exception as e:
            print(f"   ❌ Error loading checkpoint: {e}")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   You need to train a model or download a pretrained one.")

def create_test_model():
    """Create a simple test model to verify pipeline"""
    print("\n" + "=" * 60)
    print("CREATING TEST MODEL")
    print("=" * 60)
    
    import torch.nn as nn
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.fc = nn.Linear(16 * 222 * 222, 2)
        
        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    test_model = TestModel().to(DEVICE)
    test_model.eval()
    
    # Test with random input
    test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = test_model(test_input)
        probs = torch.softmax(output, dim=1)
        print(f"Test model output: Real={probs[0][0]:.3f}, Fake={probs[0][1]:.3f}")
    
    print("\n✅ Test model created successfully")
    print("   This confirms the pipeline works, but the actual model needs training.")

if __name__ == "__main__":
    print("\n🔍 DEEPFAKE MODEL DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check model file
    check_model_file()
    
    # Test model weights
    try:
        test_model_weights()
    except Exception as e:
        print(f"\n❌ Error testing weights: {e}")
    
    # Test model outputs
    try:
        test_model_outputs()
    except Exception as e:
        print(f"\n❌ Error testing outputs: {e}")
    
    # Test with an image if provided
    import sys
    if len(sys.argv) > 1:
        test_with_real_image(sys.argv[1])
    else:
        print("\n" + "=" * 60)
        print("To test with an image, run:")
        print("python diagnose_model.py path/to/image.jpg")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)