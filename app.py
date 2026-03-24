import os
import shutil
import tempfile
import streamlit as st
import cv2
import numpy as np
from predict_face import predict_image

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(page_title="Deepfake Detector", layout="wide")

# ==========================
# CUSTOM CSS (FIXED VISIBILITY)
# ==========================

st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

/* File uploader container */
[data-testid="stFileUploader"] {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
}

/* File uploader text */
[data-testid="stFileUploader"] label {
    color: white !important;
    font-size: 16px;
    font-weight: 500;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background-color: #6366f1 !important;
    color: white !important;
    border-radius: 10px;
    border: none;
}

/* Fix invisible text inside uploader */
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: white !important;
}

/* Titles */
h1, h2, h3 {
    color: #0f172a !important;
}

/* Remove small help text */
[data-testid="stFileUploader"] small {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE
# ==========================

st.title("🧠 Deepfake Detection System")
st.write("Upload an image or video to detect whether it is real or fake.")

# ==========================
# FILE UPLOAD
# ==========================

st.markdown("## Upload File")
uploaded_file = st.file_uploader("Upload file", label_visibility="collapsed")

if uploaded_file is None:
    st.info("👆 Please upload a file")
    st.stop()

# ==========================
# SAVE FILE TO TEMP
# ==========================

file_bytes = uploaded_file.read()

tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(file_bytes)
tfile.close()

# ==========================
# CLEANUP HELPER
# ==========================

def cleanup_crops():
    crops_dir = "cropped_faces"
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir)
        os.makedirs(crops_dir, exist_ok=True)

# ==========================
# CHECK FILE TYPE
# ==========================

img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
is_image = img is not None

# ==========================
# IMAGE MODE
# ==========================

try:
    if is_image:

        col1, col2 = st.columns(2)

        with col1:
            st.image(file_bytes, use_container_width=True)

        with st.spinner("Analyzing image..."):
            try:
                result = predict_image(tfile.name)

                label = result["label"]
                confidence = result["confidence"]
                faces = result["faces"]
                boxes = result["boxes"]
                face_results = result["face_results"]

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()

        if len(faces) == 0:
            st.warning("⚠️ No face detected, using full image")

        if faces:
            st.subheader("Processed Input to Model")
            for face_path in faces:
                face_img = cv2.imread(face_path) if isinstance(face_path, str) else face_path
                if face_img is not None:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    st.image(face_img, use_container_width=True)

        with col2:
            st.markdown("## Verdict")

            if "Fake" in label:
                st.error(f"🚨 Fake ({confidence:.2f}%)")
            elif label in ("Uncertain", "Low Quality / Uncertain"):
                st.warning(f"⚠️ Uncertain ({confidence:.2f}%)")
            elif label == "Invalid Image":
                st.warning("⚠️ Invalid Image — could not process")
            else:
                st.success(f"✅ Real ({confidence:.2f}%)")

            st.progress(min(confidence / 100, 1.0))

        # ==========================
        # DRAW BOXES
        # ==========================

        if img is not None and boxes:
            for i, box in enumerate(boxes):
                x, y, w, h = box

                if i < len(face_results):
                    label_text, conf = face_results[i]
                else:
                    label_text, conf = "Unknown", 0

                name = label_text.split()[1] if len(label_text.split()) > 1 else label_text

                color = (0, 0, 255) if "Fake" in label_text else (0, 255, 0)
                text = f"{name} {conf:.1f}%"

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            st.image(img, channels="BGR", use_container_width=True)

        # ==========================
        # FACE GRID
        # ==========================

        if faces:
            st.subheader("Detected Faces")
            cols = st.columns(min(len(faces), 5))

            for i, face_path in enumerate(faces):
                face_img = cv2.imread(face_path) if isinstance(face_path, str) else face_path
                if face_img is None:
                    continue
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                with cols[i % 5]:
                    st.image(face_img, caption=f"Face {i + 1}")

        cleanup_crops()

    # ==========================
    # VIDEO MODE
    # ==========================

    else:
        st.video(file_bytes)

        cap = cv2.VideoCapture(tfile.name)
        frame_results = []
        frame_count = 0

        st.write("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count > 200:
                break

            if frame_count % 10 == 0:

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as ftmp:
                    frame_path = ftmp.name

                cv2.imwrite(frame_path, frame)

                try:
                    result = predict_image(frame_path)
                    label = result["label"]
                    conf = result["confidence"]

                    frame_results.append(conf if "Fake" in label else (100 - conf))

                except Exception as e:
                    print("Frame error:", e)

                finally:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                    cleanup_crops()

        cap.release()

        # ==========================
        # FINAL VIDEO VERDICT
        # ==========================

        if frame_results:
            avg_conf = sum(frame_results) / len(frame_results)

            st.markdown("## Verdict")

            if avg_conf > 50:
                st.error(f"🚨 Fake Video ({avg_conf:.2f}%)")
            else:
                st.success(f"✅ Real Video ({100 - avg_conf:.2f}%)")

            st.progress(min(avg_conf / 100, 1.0))

        else:
            st.warning("No faces detected in video")

finally:
    try:
        os.unlink(tfile.name)
    except Exception:
        pass