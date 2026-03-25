"""
diagnose.py — Run this FIRST if the model gives wrong predictions.

Tells you exactly what is wrong with the checkpoint so you know which
fix to apply. Takes under 10 seconds.

Usage:
    python diagnose.py --real path/to/real_face.jpg \
                       --fake path/to/fake_face.jpg

Both images must be clear, unambiguous face photos.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_resnet18_best.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# BUILD MODEL (must match predictor.py exactly)
# ==========================

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Run the app once to download it, or place it manually.")
    sys.exit(1)

raw = torch.load(MODEL_PATH, map_location=DEVICE)
for k in ("state_dict", "model_state_dict", "model"):
    if isinstance(raw, dict) and k in raw:
        raw = raw[k]; break
state = {k.replace("module.", ""): v for k, v in raw.items()}
model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# HELPERS
# ==========================

def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Cannot read {path}")
        sys.exit(1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def raw_output(img: np.ndarray):
    t = transform(cv2.resize(img, (224, 224))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.softmax(logits, dim=1)
    return logits[0].cpu().numpy(), probs[0].cpu().numpy()


# ==========================
# DIAGNOSIS
# ==========================

parser = argparse.ArgumentParser()
parser.add_argument("--real", required=True)
parser.add_argument("--fake", required=True)
args = parser.parse_args()

real_logits, real_probs = raw_output(load_rgb(args.real))
fake_logits, fake_probs = raw_output(load_rgb(args.fake))

SEP = "=" * 60
print(f"\n{SEP}\nDIAGNOSIS REPORT\n{SEP}")

print(f"\nKNOWN REAL → {os.path.basename(args.real)}")
print(f"  Logits  : [{real_logits[0]:.4f},  {real_logits[1]:.4f}]")
print(f"  Softmax : [{real_probs[0]:.4f},  {real_probs[1]:.4f}]")
print(f"  Argmax  : index {real_probs.argmax()}  "
      f"({'✓ correct' if real_probs.argmax()==0 else '✗ WRONG'} — expect 0=Real)")

print(f"\nKNOWN FAKE → {os.path.basename(args.fake)}")
print(f"  Logits  : [{fake_logits[0]:.4f},  {fake_logits[1]:.4f}]")
print(f"  Softmax : [{fake_probs[0]:.4f},  {fake_probs[1]:.4f}]")
print(f"  Argmax  : index {fake_probs.argmax()}  "
      f"({'✓ correct' if fake_probs.argmax()==1 else '✗ WRONG'} — expect 1=Fake)")

print(f"\n{'-'*60}\nINTERPRETATION\n{'-'*60}")

r_win = real_probs.argmax()
f_win = fake_probs.argmax()
r_max = real_probs.max()
f_max = fake_probs.max()

if r_win == f_win and max(r_max, f_max) > 0.80:
    print("\n[PROBLEM] COLLAPSED DECISION BOUNDARY")
    print("  Both images produce the same high-confidence class.")
    print("  The model never learned to separate real from fake.")
    print("\n  CAUSE : The Navxx checkpoint converges to a constant output.")
    print("  FIX 1 : Use a different, verified checkpoint.")
    print("  FIX 2 : Retrain using finetune.py on labeled data.")
    print("  FIX 3 : Try bias-correction — add a per-class bias offset to")
    print("          the final linear layer to shift the decision boundary.")

elif r_win == 1 and f_win == 0:
    print("\n[PROBLEM] SWAPPED CLASS INDICES")
    print("  Real image wins on index 1 (labeled Fake in code).")
    print("  Fake image wins on index 0 (labeled Real in code).")
    print("\n  FIX : In predictor.py swap the index references:")
    print("        real_p = prob[1].item()   # was prob[0]")
    print("        fake_p = prob[0].item()   # was prob[1]")
    print("        And in CLASS_NAMES: [\"Human Fake Face\", \"Human Real Face\"]")

elif r_win == 0 and f_win == 1:
    if real_probs[0] >= 0.65 and fake_probs[1] >= 0.65:
        print("\n[OK] Class indices are correct AND confidence is strong.")
        print("  real_prob[0] = {:.3f}  (Real image correctly → index 0)".format(real_probs[0]))
        print("  fake_prob[1] = {:.3f}  (Fake image correctly → index 1)".format(fake_probs[1]))
        print("\n  The model looks healthy. If you still see wrong predictions,")
        print("  the problem is in preprocessing or face detection, not the model.")
        print("  Run: python -c \"from predictor import predict_image; "
              "print(predict_image('your_image.jpg'))\"")
    else:
        print("\n[WARN] Class indices correct but confidence is LOW.")
        print(f"  real_prob[0] = {real_probs[0]:.3f}  (want ≥ 0.65)")
        print(f"  fake_prob[1] = {fake_probs[1]:.3f}  (want ≥ 0.65)")
        print("\n  The model outputs the right class but is not confident.")
        print("  This usually means the checkpoint is undertrained.")
        print("  FIX: Lower FAKE_THRESHOLD to 0.50 and REAL_THRESHOLD to 0.40")
        print("       in predictor.py to reduce the uncertain band,")
        print("       or fine-tune the model on more labeled data.")

else:
    print("\n[PROBLEM] INCONSISTENT / RANDOM PREDICTIONS")
    print(f"  Real image → index {r_win} wins  (want 0)")
    print(f"  Fake image → index {f_win} wins  (want 1)")
    print("\n  The model is not reliably separating the two classes.")
    print("  FIX: Replace the checkpoint with a verified one.")

# Logit spread
spread_r = abs(real_logits[0] - real_logits[1])
spread_f = abs(fake_logits[0] - fake_logits[1])
print(f"\nLogit spread — real image : {spread_r:.3f}")
print(f"Logit spread — fake image : {spread_f:.3f}")
if spread_r < 0.5 and spread_f < 0.5:
    print("WARNING: Both images show very small logit spread.")
    print("  This confirms the model has no meaningful decision boundary.")

print(f"\n{SEP}")
print("Next step: if the model is collapsed, run predictor.py with a")
print("different checkpoint, or retrain with finetune.py.")
print(f"{SEP}\n")