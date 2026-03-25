"""
app.py — Streamlit UI for deepfake detection.

Imports:
    predictor.py       — predict_image()
    video_predictor.py — predict_video(), fake_probability_timeline()
"""

import os
import shutil
import tempfile
import streamlit as st
import cv2
import numpy as np

from predict_face import predict_image
from predict_video import predict_video, fake_probability_timeline

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f8fafc; color: #0f172a; }
[data-testid="stFileUploader"] {
    background-color: #1e293b; padding: 20px; border-radius: 15px;
}
[data-testid="stFileUploader"] label  { color: white !important; font-size: 16px; font-weight: 500; }
[data-testid="stFileUploader"] button { background-color: #6366f1 !important; color: white !important; border-radius: 10px; border: none; }
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p      { color: white !important; }
[data-testid="stFileUploader"] small  { display: none; }
h1, h2, h3 { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE
# ==========================

st.title("Deepfake Detection System")
st.write("Upload an image or video to detect whether it is real or fake.")

# ==========================
# FILE UPLOAD
# ==========================

st.markdown("## Upload File")
uploaded_file = st.file_uploader(
    "Upload file",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("Please upload a file to begin.")
    st.stop()

# ==========================
# SAVE FILE TO TEMP
# ==========================

file_bytes = uploaded_file.getvalue()
suffix     = os.path.splitext(uploaded_file.name)[1] or ".tmp"
tfile      = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tfile.write(file_bytes)
tfile.close()

# ==========================
# HELPERS
# ==========================

def cleanup_crops():
    crops_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cropped_faces"
    )
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir)
    os.makedirs(crops_dir, exist_ok=True)


def show_verdict(label: str, confidence: float):
    """Render verdict box + progress bar consistently for both image and video."""
    st.markdown("## Verdict")
    if "Fake" in label:
        st.error(f"🚨 Fake — {confidence:.1f}% confidence")
    elif label in ("Uncertain", "Low Quality / Uncertain"):
        st.warning(f"⚠️ Uncertain — {confidence:.1f}% confidence")
    elif "Invalid" in label:
        st.warning("⚠️ Could not process this file")
    else:
        st.success(f"✅ Real — {confidence:.1f}% confidence")
    st.progress(min(confidence / 100.0, 1.0))


# ==========================
# DETECT FILE TYPE
# ==========================

img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
is_image  = img_array is not None

try:

    # ==========================
    # IMAGE MODE
    # ==========================

    if is_image:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Uploaded Image")
            st.image(file_bytes, use_container_width=True)

        with st.spinner("Analysing image…"):
            try:
                result = predict_image(tfile.name)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        label        = result["label"]
        confidence   = result["confidence"]
        faces        = result["faces"]
        boxes        = result["boxes"]
        face_results = result["face_results"]

        with col2:
            show_verdict(label, confidence)

            if not faces:
                st.info("No face detected — result is based on full image "
                        "(reliability is lower without a face crop).")

            # Per-face breakdown
            if face_results:
                st.markdown("### Per-Face Results")
                for i, fr in enumerate(face_results):
                    fl, fc = fr[0], fr[1]
                    emoji  = "🚨" if "Fake" in fl else "✅"
                    st.write(f"Face {i+1}: {emoji} {fl} ({fc:.1f}%)")

        # ------------------------------------------------------------------
        # ANNOTATED IMAGE WITH BOUNDING BOXES
        # ------------------------------------------------------------------
        if img_array is not None and boxes:
            annotated = img_array.copy()
            for i, (x, y, w, h) in enumerate(boxes):
                fl, fc = (face_results[i][0], face_results[i][1]) \
                          if i < len(face_results) else ("Unknown", 0.0)
                color = (0, 0, 255) if "Fake" in fl else (0, 200, 0)
                parts = fl.split()
                short = parts[1] if len(parts) > 1 else fl
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, f"{short} {fc:.1f}%",
                            (x, max(y - 10, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            st.markdown("### Detection Map")
            st.image(annotated, channels="BGR", use_container_width=True)

        # ------------------------------------------------------------------
        # GRADCAM HEATMAP GRID
        # ------------------------------------------------------------------
        if faces:
            st.markdown("### GradCAM — where the model looked")
            cols = st.columns(max(len(faces), 1))

            for i, face_path in enumerate(faces):
                face_bgr = cv2.imread(face_path) if isinstance(face_path, str) \
                           else face_path
                if face_bgr is None:
                    continue

                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                # face_results entries are (label, confidence, heatmap)
                if i < len(face_results) and len(face_results[i]) >= 3:
                    fl, fc, heatmap = face_results[i]
                else:
                    fl, fc, heatmap = "Unknown", 0.0, None

                # Show heatmap for uncertain/fake faces; plain crop for clear reals
                show_heatmap = (heatmap is not None) and \
                               ("Fake" in fl or fc < 70.0)

                if show_heatmap:
                    hmap_colored = cv2.applyColorMap(
                        np.uint8(255 * heatmap), cv2.COLORMAP_JET
                    )
                    hmap_colored = cv2.cvtColor(hmap_colored, cv2.COLOR_BGR2RGB)
                    hmap_colored = cv2.resize(
                        hmap_colored, (face_rgb.shape[1], face_rgb.shape[0])
                    )
                    overlay = cv2.addWeighted(face_rgb, 0.6, hmap_colored, 0.4, 0)
                    caption = f"Face {i+1}: suspicion zones highlighted"
                else:
                    overlay = face_rgb
                    caption = f"Face {i+1}: confirmed real"

                with cols[i]:
                    st.image(overlay, caption=caption, use_container_width=True)

        cleanup_crops()

    # ==========================
    # VIDEO MODE
    # ==========================

    else:
        st.video(file_bytes)
        st.markdown("## Analysing Video…")

        progress_bar = st.progress(0)
        status_text  = st.empty()

        def on_progress(fraction: float, status: str):
            progress_bar.progress(min(fraction, 1.0))
            status_text.text(status)

        with st.spinner("Running deepfake analysis on video frames…"):
            result = predict_video(tfile.name, progress_callback=on_progress)

        progress_bar.progress(1.0)
        status_text.empty()

        # Show any non-fatal warning
        if result.get("error"):
            st.warning(f"⚠️ {result['error']}")

        # ------------------------------------------------------------------
        # MAIN VERDICT
        # ------------------------------------------------------------------
        show_verdict(result["label"], result["confidence"])

        st.caption(
            f"Based on **{result['frames_used']}** analysed frame(s)  •  "
            f"Mean fake probability: **{result['fake_prob']:.3f}**"
        )

        # ------------------------------------------------------------------
        # TIMELINE CHART
        # ------------------------------------------------------------------
        frame_log = result.get("frame_log", [])
        if len(frame_log) >= 2:
            st.markdown("### Fake Probability Timeline")
            st.caption(
                "Each point is one sampled frame.  "
                "Above 0.55 = likely fake  |  Below 0.45 = likely real  |  "
                "Between = uncertain."
            )

            import pandas as pd
            indices, probs = fake_probability_timeline(frame_log)
            chart_df = pd.DataFrame({"Fake Probability": probs}, index=indices)
            chart_df.index.name = "Frame"
            st.line_chart(chart_df)

            # Threshold reference lines via caption (st.line_chart has no overlay)
            st.caption(
                "Reference: fake threshold = 0.55 (red), real threshold = 0.45 (green)"
            )

            # Per-frame detail table (collapsed)
            with st.expander("📋 Per-frame detail"):
                rows = []
                for entry in frame_log:
                    fp    = entry["fake_prob"]
                    emoji = "🚨" if fp >= 0.55 else ("⚠️" if fp >= 0.45 else "✅")
                    rows.append({
                        "Frame":     entry["frame_idx"],
                        "Result":    f"{emoji} {'Fake' if fp>=0.55 else ('Uncertain' if fp>=0.45 else 'Real')}",
                        "Fake Prob": f"{fp:.3f}",
                    })
                st.dataframe(rows, use_container_width=True)

# ==========================
# ALWAYS delete temp upload file
# ==========================
finally:
    try:
        os.unlink(tfile.name)
    except Exception:
        pass