import streamlit as st
import tempfile
import cv2
import numpy as np
from predict_face import predict_image

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(page_title="Deepfake Detector", layout="wide")

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}

/* Upload box styling */
[data-testid="stFileUploader"] {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stFileUploader"] label {
    color: white;
    font-size: 16px;
}

[data-testid="stFileUploader"] button {
    background-color: #6366f1;
    color: white;
    border-radius: 10px;
}

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
uploaded_file = st.file_uploader("", label_visibility="collapsed")

if uploaded_file is None:
    st.info("👆 Please upload a file")
    st.stop()

# ==========================
# SAVE FILE
# ==========================

file_bytes = uploaded_file.read()

tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(file_bytes)
tfile.close()

# ==========================
# CHECK TYPE
# ==========================

img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
is_image = img is not None

# ==========================
# IMAGE MODE
# ==========================

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

    with col2:
        st.markdown("## Verdict")

        if "Fake" in label:
            st.error(f"🚨 Fake ({confidence:.2f}%)")
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

            label_parts = label_text.split()
            name = label_parts[1] if len(label_parts) > 1 else label_text

            color = (0, 0, 255) if "Fake" in label_text else (0, 255, 0)
            text = f"{name} {conf:.1f}%"

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
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
            if isinstance(face_path, str):
                face_img = cv2.imread(face_path)
            else:
                face_img = face_path

            if face_img is None:
                continue

            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            with cols[i % 5]:
                st.image(face_img, caption=f"Face {i+1}")

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

        if frame_count % 10 == 0:

            frame_path = f"frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)

            try:
                result = predict_image(frame_path)

                label = result["label"]
                conf = result["confidence"]

                if "Fake" in label:
                    frame_results.append(conf)
                else:
                    frame_results.append(100 - conf)

            except:
                pass

        frame_count += 1

    cap.release()

    # ==========================
    # FINAL RESULT
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
        st.warning("No faces detected")