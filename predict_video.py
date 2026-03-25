"""
video_predictor.py — Video deepfake predictor.

Samples frames at ~1 fps, runs face detection + TTA inference on each,
then aggregates using a weighted mean + median combination that is robust
to occasional bad frames (motion blur, partial faces, occlusion).

Usage (CLI):
    python video_predictor.py <video_path>

Depends on:
    predictor.py    — predict_with_tta, IMG_SIZE, FAKE_THRESHOLD, REAL_THRESHOLD
    face_cropper.py — detect_faces_dnn
    utils.py        — enhance_image, gentle_sharpen, laplacian_variance
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

from face_cropper import detect_faces_dnn
from utils import enhance_image, gentle_sharpen, laplacian_variance
from predict_face import predict_with_tta, IMG_SIZE, FAKE_THRESHOLD, REAL_THRESHOLD

# ==========================
# CONFIGURATION
# ==========================

MAX_SAMPLE_FRAMES      = 60     # hard cap on frames analysed
MIN_FRAMES_FOR_VERDICT = 3      # below this → Uncertain (not enough signal)
MIN_FACE_PX            = 40     # skip detections smaller than this
BLUR_THRESHOLD         = 30     # skip blurry frames
MIN_FRAME_CONFIDENCE   = 0.45   # skip near-random model outputs (p < 0.45)
SAMPLE_EVERY_N_SECONDS = 1.0    # target sampling rate

# Video-specific class names (distinct from image-level CLASS_NAMES)
VIDEO_CLASS_NAMES = ["Real Video", "Fake Video"]


# ==========================
# HELPERS
# ==========================

def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(val, hi))


def _crop_face_from_box(
    rgb_frame: np.ndarray,
    box: list,
    margin: float = 0.20,
) -> np.ndarray:
    """
    Crop a detected face with proportional context margins.

    Args:
        rgb_frame : full-resolution RGB uint8 frame.
        box       : [x, y, w, h] from detect_faces_dnn.
        margin    : fraction of face size to add on each side.
    Returns:
        RGB uint8 crop, or empty array if coordinates are degenerate.
    """
    h_img, w_img = rgb_frame.shape[:2]
    x, y, w, h   = box
    mx = int(w * margin)
    my = int(h * margin)
    x1 = _clamp(x - mx,         0, w_img)
    x2 = _clamp(x + w + mx,     0, w_img)
    y1 = _clamp(y - my,          0, h_img)
    y2 = _clamp(y + h + my,      0, h_img)
    return rgb_frame[y1:y2, x1:x2]


def _sample_frame_indices(total_frames: int, fps: float) -> List[int]:
    """
    Build a list of frame indices to sample.

    Samples ~1 frame/second throughout the video.
    Always includes first and last frames for context.
    Caps at MAX_SAMPLE_FRAMES, distributed evenly.
    """
    if total_frames <= 0:
        return []

    step    = max(1, int(round(fps * SAMPLE_EVERY_N_SECONDS)))
    indices = list(range(0, total_frames, step))

    # Sub-sample evenly if too many
    if len(indices) > MAX_SAMPLE_FRAMES:
        chosen  = np.linspace(0, len(indices) - 1, MAX_SAMPLE_FRAMES, dtype=int)
        indices = [indices[i] for i in chosen]

    # Always include first and last
    for special in (0, total_frames - 1):
        if special not in indices:
            indices.append(special)

    return sorted(set(indices))


def _result_to_fake_prob(label: str, confidence: float) -> float:
    """
    Convert a predict_image() result to a fake probability in [0, 1].

    This is the function that fixes the original app.py video bug where:
      - Fake  → conf used directly as a percentage → correct scale but
      - Real  → 100 - conf appended → this IS correct (converts real-
                confidence to a fake score on the same 0-100 scale)

    However, accumulating as 0-100 and comparing to 50 is numerically
    identical to 0-1 compared to 0.5, so the original logic was actually
    fine for the accumulation step.

    The real bug was that app.py mixed 'conf' (0-100) and 'fake_score'
    (also 0-100) without distinguishing them — here we use a single
    consistent [0, 1] fake probability throughout.

      "Fake"      → confidence / 100       e.g. 82% fake  → 0.82
      "Real"      → 1 - confidence / 100   e.g. 90% real  → 0.10
      "Uncertain" → 0.5                    neutral, no signal
    """
    if "Fake" in label:
        return confidence / 100.0
    elif "Real" in label:
        return 1.0 - (confidence / 100.0)
    else:
        return 0.5


def _build_verdict(mean_fake_prob: float, n_frames: int) -> Dict:
    """Convert a mean fake probability → verdict dict."""
    if mean_fake_prob >= FAKE_THRESHOLD:
        label = VIDEO_CLASS_NAMES[1]            # "Fake Video"
        conf  = round(min(mean_fake_prob * 100, 100.0), 2)
    elif mean_fake_prob <= REAL_THRESHOLD:
        label = VIDEO_CLASS_NAMES[0]            # "Real Video"
        conf  = round(min((1.0 - mean_fake_prob) * 100, 100.0), 2)
    else:
        label = "Uncertain"
        conf  = round(abs(mean_fake_prob - 0.5) * 200, 2)

    return {
        "label":        label,
        "confidence":   conf,
        "fake_prob":    round(mean_fake_prob, 4),
        "frames_used":  n_frames,
    }


# ==========================
# PER-FRAME INFERENCE
# ==========================

def _analyse_frame(rgb_frame: np.ndarray) -> Optional[float]:
    """
    Run face detection + TTA inference on a single RGB frame.

    Returns the fake probability (0-1) of the most confident face,
    or None if the frame has no usable face.

    Using the most confident face (rather than averaging all faces)
    is safer for video — a single clearly fake face in a crowd is
    the important signal, not an average diluted by real bystanders.
    """
    detections = detect_faces_dnn(rgb_frame)
    if not detections:
        return None

    best_fake_p: Optional[float] = None
    best_conf   = -1.0

    for det in detections:
        face = _crop_face_from_box(rgb_frame, det["box"])
        if face.size == 0:
            continue
        if face.shape[0] < MIN_FACE_PX or face.shape[1] < MIN_FACE_PX:
            continue

        # Enhance / sharpen on full-res crop BEFORE resize
        if np.std(face) < 50:
            face = enhance_image(face)
        else:
            face = gentle_sharpen(face)

        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        if laplacian_variance(face_resized) < BLUR_THRESHOLD:
            continue

        prob   = predict_with_tta(face_resized)
        real_p = prob[0].item()
        fake_p = prob[1].item()

        if max(real_p, fake_p) < MIN_FRAME_CONFIDENCE:
            continue

        # Keep the face with the highest detector confidence
        if det["confidence"] > best_conf:
            best_conf   = det["confidence"]
            best_fake_p = fake_p

    return best_fake_p


# ==========================
# MAIN FUNCTION
# ==========================

def predict_video(
    video_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict:
    """
    Analyse a video file for deepfake content.

    Args:
        video_path        : path to the video file.
        progress_callback : optional callable(fraction, status_string).
                            fraction is in [0, 1].

    Returns dict:
        label        — verdict string (VIDEO_CLASS_NAMES entry or "Uncertain")
        confidence   — float 0-100
        fake_prob    — raw combined fake probability (0-1)
        frames_used  — number of frames that contributed
        frame_log    — list of per-frame dicts for charting:
                         { frame_idx, fake_prob }
        error        — None on success, error string on failure
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "label": "Invalid Video", "confidence": 0.0,
            "fake_prob": 0.5, "frames_used": 0,
            "frame_log": [], "error": f"Cannot open: {video_path}",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_indices = _sample_frame_indices(total_frames, fps)
    n_to_sample    = len(sample_indices)

    print(f"\n[video_predictor] {os.path.basename(video_path)}")
    print(f"  total_frames={total_frames}  fps={fps:.1f}  "
          f"sampling {n_to_sample} frame(s)")

    fake_probs: List[float] = []
    frame_log:  List[Dict]  = []

    for sample_num, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  [skip] frame {frame_idx} — could not read")
            continue

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fake_p   = _analyse_frame(rgb)

        if fake_p is not None:
            fake_probs.append(fake_p)
            frame_log.append({"frame_idx": frame_idx, "fake_prob": round(fake_p, 4)})
            print(f"  frame {frame_idx:5d}  fake_p={fake_p:.3f}")
        else:
            print(f"  frame {frame_idx:5d}  — no usable face")

        if progress_callback is not None:
            fraction = (sample_num + 1) / max(n_to_sample, 1)
            progress_callback(
                min(fraction, 1.0),
                f"Analysed {len(fake_probs)} frame(s) with faces…"
            )

    cap.release()

    if len(fake_probs) < MIN_FRAMES_FOR_VERDICT:
        return {
            "label": "Uncertain", "confidence": 0.0,
            "fake_prob": 0.5, "frames_used": len(fake_probs),
            "frame_log": frame_log,
            "error": (
                f"Only {len(fake_probs)} usable frame(s) found "
                f"(need ≥ {MIN_FRAMES_FOR_VERDICT})."
            ),
        }

    arr       = np.array(fake_probs)

    # Rolling smoothing — reduces impact of single wildly-wrong frames.
    # mode="same" keeps array length constant (unlike "valid" which shrinks it).
    window   = min(5, len(arr))
    kernel   = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")

    mean_p   = float(np.mean(smoothed))
    median_p = float(np.median(smoothed))

    # 70% mean + 30% median: mean captures overall signal,
    # median resists outlier frames that are strongly mispredicted.
    combined = 0.70 * mean_p + 0.30 * median_p

    print(f"\n[video_predictor] mean={mean_p:.3f}  median={median_p:.3f}  "
          f"combined={combined:.3f}  frames_used={len(fake_probs)}")

    verdict = _build_verdict(combined, len(fake_probs))
    verdict["frame_log"] = frame_log
    verdict["error"]     = None
    return verdict


# ==========================
# TIMELINE UTILITY
# ==========================

def fake_probability_timeline(
    frame_log: List[Dict],
) -> Tuple[List[int], List[float]]:
    """
    Extract parallel (frame_indices, fake_probs) from a frame_log.
    Useful for plotting a timeline chart in the UI.
    """
    indices = [e["frame_idx"]  for e in frame_log]
    probs   = [e["fake_prob"]  for e in frame_log]
    return indices, probs


# ==========================
# CLI ENTRY POINT
# ==========================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_predictor.py <video_path>")
        sys.exit(1)

    result = predict_video(sys.argv[1])

    print(f"\n{'='*45}")
    print("  FINAL VIDEO VERDICT")
    print(f"{'='*45}")
    print(f"  Prediction  : {result['label']}")
    print(f"  Confidence  : {result['confidence']:.2f}%")
    print(f"  Frames used : {result['frames_used']}")
    print(f"  Mean fake p : {result['fake_prob']:.4f}")
    if result.get("error"):
        print(f"  Note        : {result['error']}")
    print(f"{'='*45}\n")