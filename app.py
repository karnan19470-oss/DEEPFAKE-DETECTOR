"""
app.py — Complete app without matplotlib dependency
"""

import os
import sys
import shutil
import tempfile
import traceback
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Define cleanup function at the very beginning
def cleanup_crops():
    """Clean up cropped faces directory"""
    crops_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cropped_faces")
    if os.path.exists(crops_dir):
        try:
            shutil.rmtree(crops_dir)
        except:
            pass
    os.makedirs(crops_dir, exist_ok=True)

# Try to import predict_face with error handling
try:
    from predict_face import predict_image, FAKE_THRESHOLD
    print("Successfully imported predict_face")
except Exception as e:
    st.error(f"Failed to import predict_face: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Try to import predict_video with error handling
try:
    from predict_video import predict_video
    print("Successfully imported predict_video")
except Exception as e:
    st.warning(f"Video prediction unavailable: {e}")
    predict_video = None

# Try to import utils with error handling
try:
    from utils import laplacian_variance, estimate_blockiness
    print("Successfully imported utils")
except Exception as e:
    st.warning(f"Utils import failed: {e}")
    # Provide fallback functions
    def laplacian_variance(img):
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except:
            return 25.0
    
    def estimate_blockiness(gray):
        return 0.0

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .verdict-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .verdict-real {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    .verdict-error {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .confidence-bar {
        background-color: #e5e7eb;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #10b981 0%, #ef4444 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'model_status' not in st.session_state:
    st.session_state.model_status = "Unknown"

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Settings")
    
    # Model status indicator
    st.markdown("### 🤖 Model Status")
    try:
        from predict_face import model
        if model is not None:
            st.success("✅ Model loaded successfully")
            st.session_state.model_status = "Loaded"
        else:
            st.error("❌ Model failed to load")
            st.session_state.model_status = "Failed"
    except:
        st.error("❌ Model import failed")
        st.session_state.model_status = "Failed"
    
    st.markdown("---")
    
    # Analysis mode
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Standard", "Detailed", "Debug"],
        help="Standard: Quick analysis, Detailed: Full metrics, Debug: Show model internals"
    )
    
    if analysis_mode == "Debug":
        st.session_state.debug_mode = True
    else:
        st.session_state.debug_mode = False
    
    st.markdown("---")
    
    # Confidence threshold
    custom_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.3,
        max_value=0.8,
        value=0.55,
        step=0.01,
        help="Adjust sensitivity of fake detection (lower = more sensitive)"
    )
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        enable_tta = st.checkbox("Test-Time Augmentation", value=True, help="Improves accuracy but slower")
        enable_ensemble = st.checkbox("Ensemble Prediction", value=True, help="Use multiple predictions")
        enable_quality_check = st.checkbox("Quality Check", value=True, help="Filter low-quality inputs")
    
    st.markdown("---")
    
    # History
    if st.session_state.analysis_history:
        st.markdown("## 📊 Recent Analyses")
        for item in st.session_state.analysis_history[-5:]:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{item['filename'][:20]}**")
                    st.write(f"*{item['timestamp']}*")
                with col2:
                    if "Fake" in item['label']:
                        st.error(f"🚨 {item['confidence']:.1f}%")
                    else:
                        st.success(f"✅ {item['confidence']:.1f}%")
                st.markdown("---")
        
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Info
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Deepfake Detection System v2.0**
        
        - Uses ResNet18 architecture
        - Ensemble predictions for accuracy
        - Real-time face detection
        - Quality assessment filters
        
        **Note:** Results should be interpreted with caution.
        No system is 100% accurate.
        """)

# Main header
st.markdown('<div class="main-header"><h1>🔍 Deepfake Detection System</h1><p>Advanced AI-powered fake media detection</p></div>', unsafe_allow_html=True)

# Show warning if model failed to load
if st.session_state.model_status == "Failed":
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Model Not Loaded</strong><br>
        The deepfake detection model could not be loaded. Predictions may be inaccurate or unavailable.
        Please check that the model file 'deepfake_resnet18_best.pth' exists in the application directory.
    </div>
    """, unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv", "webm"],
    help="Supported formats: Images (JPG, PNG, BMP) and Videos (MP4, AVI, MOV)"
)

if uploaded_file is None:
    # Show placeholder with instructions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📸 **Image Analysis**\n\nUpload photos to detect manipulated faces")
    with col2:
        st.info("🎬 **Video Analysis**\n\nAnalyze videos frame-by-frame for deepfake patterns")
    with col3:
        st.info("🔬 **Advanced Metrics**\n\nView detailed quality metrics and confidence scores")
    st.stop()

# Process file
file_bytes = uploaded_file.getvalue()
suffix = os.path.splitext(uploaded_file.name)[1] or ".tmp"
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tfile.write(file_bytes)
tfile.close()

# Determine if image or video
try:
    img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    is_image = img_array is not None
except:
    is_image = False

try:
    if is_image:
        # ==========================
        # IMAGE ANALYSIS
        # ==========================
        
        # Display original image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(file_bytes, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Image Information")
            h, w = img_array.shape[:2]
            st.write(f"**Resolution:** {w} x {h}")
            st.write(f"**File size:** {len(file_bytes) / 1024:.1f} KB")
            st.write(f"**Format:** {suffix.upper()}")
            
            # Quick quality metrics
            try:
                blur_score = laplacian_variance(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                contrast = np.std(img_array)
                
                with st.expander("📈 Quick Quality Check"):
                    st.write(f"**Blur Score:** {blur_score:.1f} {'(Blurry)' if blur_score < 25 else '(Sharp)'}")
                    st.write(f"**Contrast:** {contrast:.1f} {'(Low)' if contrast < 30 else '(Good)'}")
                    
                    # Simple quality bar
                    quality_score = min(100, (blur_score/50)*50 + (contrast/100)*50)
                    st.write(f"**Quality Score:** {quality_score:.0f}/100")
                    st.progress(quality_score/100)
            except Exception as e:
                st.warning(f"Quality check failed: {e}")
        
        # Run prediction
        with st.spinner("Analyzing image..."):
            try:
                result = predict_image(tfile.name)
                
                # Store in history
                st.session_state.analysis_history.append({
                    'filename': uploaded_file.name,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'label': result['label'],
                    'confidence': result['confidence']
                })
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                if st.session_state.debug_mode:
                    st.code(traceback.format_exc())
                st.stop()
        
        # Display verdict
        label = result["label"]
        confidence = result["confidence"]
        
        with col2:
            if "Fake" in label:
                st.markdown(f'<div class="verdict-box verdict-fake"><h2>🚨 {label}</h2><h3>{confidence:.1f}% confidence</h3></div>', unsafe_allow_html=True)
            elif "Real" in label:
                st.markdown(f'<div class="verdict-box verdict-real"><h2>✅ {label}</h2><h3>{confidence:.1f}% confidence</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-box verdict-error"><h2>⚠️ {label}</h2></div>', unsafe_allow_html=True)
        
        # Detailed metrics
        if analysis_mode in ["Detailed", "Debug"]:
            st.markdown("---")
            st.markdown("### 🔬 Detailed Analysis")
            
            # Get quality metrics
            if 'quality' in result:
                st.info(f"📈 {result['quality']}")
            
            # Display fake probability if available
            if 'fake_prob' in result:
                fake_prob = result['fake_prob']
                st.write(f"**Raw Fake Probability:** {fake_prob:.3f}")
                st.write(f"**Threshold:** {custom_threshold}")
                
                # Create confidence bar
                st.write("**Fake Probability**")
                st.progress(fake_prob)
                st.caption(f"{fake_prob*100:.1f}%")
                
                # Show which side of threshold
                if fake_prob >= custom_threshold:
                    st.error(f"⚠️ Above threshold → Fake")
                else:
                    st.success(f"✅ Below threshold → Real")
            
            # Display per-face results
            face_results = result.get("face_results", [])
            if face_results:
                st.markdown("#### 👤 Per-Face Analysis")
                
                face_data = []
                for i, (fr_label, fr_conf, _) in enumerate(face_results):
                    face_data.append({
                        "Face": f"Face {i+1}",
                        "Result": "🚨 Fake" if "Fake" in fr_label else "✅ Real",
                        "Confidence": f"{fr_conf:.1f}%"
                    })
                
                df = pd.DataFrame(face_data)
                st.dataframe(df, use_container_width=True)
        
        # Debug mode
        if st.session_state.debug_mode:
            st.markdown("---")
            st.markdown("### 🐛 Debug Information")
            
            with st.expander("Show Debug Info"):
                # Show key information
                st.write("**Prediction Details:**")
                st.json({
                    "label": result.get("label"),
                    "confidence": result.get("confidence"),
                    "faces_detected": len(result.get("faces", [])),
                    "face_results": result.get("face_results", []),
                    "error": result.get("error", False)
                })
    
    else:
        # ==========================
        # VIDEO ANALYSIS
        # ==========================
        
        if predict_video is None:
            st.error("Video analysis is unavailable. Please install required dependencies.")
            st.stop()
        
        # Display video
        st.markdown("### 🎬 Video Preview")
        st.video(file_bytes)
        
        # Video info
        try:
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{duration:.1f}s")
            with col2:
                st.metric("FPS", f"{fps:.1f}")
            with col3:
                st.metric("Total Frames", total_frames)
            with col4:
                st.metric("Resolution", f"{width}x{height}")
        except Exception as e:
            st.warning(f"Could not read video info: {e}")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def on_progress(fraction: float, status: str):
            progress_bar.progress(min(fraction, 1.0))
            status_text.text(status)
        
        # Run video prediction
        with st.spinner("Analyzing video frames..."):
            try:
                result = predict_video(tfile.name, progress_callback=on_progress)
                
                # Store in history
                st.session_state.analysis_history.append({
                    'filename': uploaded_file.name,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'label': result['label'],
                    'confidence': result['confidence']
                })
                
            except Exception as e:
                st.error(f"Video analysis error: {e}")
                if st.session_state.debug_mode:
                    st.code(traceback.format_exc())
                st.stop()
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Display verdict
        if result.get("error"):
            st.warning(result["error"])
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            label = result["label"]
            confidence = result["confidence"]
            
            if "Fake" in label:
                st.markdown(f'<div class="verdict-box verdict-fake"><h2>🚨 {label}</h2><h3>{confidence:.1f}% confidence</h3></div>', unsafe_allow_html=True)
            elif "Real" in label:
                st.markdown(f'<div class="verdict-box verdict-real"><h2>✅ {label}</h2><h3>{confidence:.1f}% confidence</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-box verdict-error"><h2>⚠️ {label}</h2></div>', unsafe_allow_html=True)
        
        with col2:
            frames_used = result.get("frames_used", 0)
            fake_prob = result.get("fake_prob", 0)
            st.metric("Frames Analyzed", frames_used)
            st.metric("Mean Fake Probability", f"{fake_prob:.3f}")
            
            # Add confidence bar
            st.write("**Fake Probability**")
            st.progress(fake_prob)
        
        # Display frame log in expander
        frame_log = result.get("frame_log", [])
        if frame_log and analysis_mode in ["Detailed", "Debug"]:
            with st.expander("📊 Frame Details"):
                # Create simple table
                frame_data = []
                for f in frame_log[-20:]:  # Show last 20 frames
                    frame_data.append({
                        "Frame": f["frame_idx"],
                        "Time (s)": f"{f['timestamp']:.1f}",
                        "Fake Prob": f"{f.get('fake_prob_smoothed', f.get('fake_prob', 0)):.3f}",
                        "Faces": f.get('num_faces', 1)
                    })
                
                df = pd.DataFrame(frame_data)
                st.dataframe(df, use_container_width=True)
        
        # Statistics summary
        if 'statistics' in result:
            with st.expander("📈 Detailed Statistics"):
                stats = result['statistics']
                st.write(f"**Mean Probability:** {stats.get('mean', 0):.3f}")
                st.write(f"**Median Probability:** {stats.get('median', 0):.3f}")
                st.write(f"**Standard Deviation:** {stats.get('std', 0):.3f}")
                st.write(f"**Frames Analyzed:** {stats.get('frames_analyzed', 0)}")
                if 'threshold_used' in result:
                    st.write(f"**Threshold Used:** {result['threshold_used']:.3f}")
        
        # Download report option
        if st.button("📥 Download Analysis Report"):
            report_data = {
                "filename": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "verdict": label,
                "confidence": confidence,
                "frames_analyzed": result.get("frames_used", 0),
                "mean_fake_probability": result.get("fake_prob", 0),
                "frame_log": frame_log,
                "statistics": result.get("statistics", {})
            }
            
            report_json = json.dumps(report_data, indent=2)
            
            st.download_button(
                label="Download JSON Report",
                data=report_json,
                file_name=f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Cleanup
    cleanup_crops()

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    if st.session_state.debug_mode:
        st.write("### Error Details")
        st.code(traceback.format_exc())

finally:
    try:
        os.unlink(tfile.name)
    except Exception:
        pass

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔍 Deepfake Detection System v2.0 | Using ResNet18 with ensemble predictions</p>
    <p>⚠️ Note: Results should be interpreted with caution. No system is 100% accurate.</p>
</div>
""", unsafe_allow_html=True)