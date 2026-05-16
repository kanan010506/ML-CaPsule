import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os

from Detection import (
    load_model,
    detect_animals,
    get_highest_danger,
    DANGER_EMOJI,
    DANGER_MAP,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Animal Detection & Alert System",
    page_icon="🐾",
    layout="wide",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .alert-HIGH   { background:#ff4444; color:white; padding:14px 18px;
                    border-radius:8px; font-size:1.1rem; font-weight:600; }
    .alert-MEDIUM { background:#ff8800; color:white; padding:14px 18px;
                    border-radius:8px; font-size:1.1rem; font-weight:600; }
    .alert-LOW    { background:#00aa44; color:white; padding:14px 18px;
                    border-radius:8px; font-size:1.1rem; font-weight:600; }
    .alert-NONE   { background:#555555; color:white; padding:14px 18px;
                    border-radius:8px; font-size:1.1rem; }
    .stat-box     { background:#1e1e2e; color:white; padding:16px;
                    border-radius:10px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.title("🐾 Animal Detection & Alert System")
st.markdown(
    "**YOLOv8 + Danger Classification** — Detects animals and raises alerts "
    "based on danger level (HIGH / MEDIUM / LOW)"
)
st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    model_variant  = st.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])

    st.markdown("---")
    st.header("📋 Danger Levels")
    for animal, level in DANGER_MAP.items():
        st.markdown(f"{DANGER_EMOJI[level]} **{animal.title()}** — {level}")

    st.markdown("---")
    st.caption("Built for ML-CaPsule · GSSoC 2026")

# ============================================================
# MODEL LOAD
# ============================================================
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def get_model(path):
    return load_model(path)

model = get_model(model_variant)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎬 Video Detection", "📹 Webcam (Live)"])


# ----------------------------------------------------------
# TAB 1 — IMAGE
# ----------------------------------------------------------
with tab1:
    st.subheader("Upload an Image")
    uploaded_img = st.file_uploader(
        "Upload image (jpg / png)", type=["jpg", "jpeg", "png"], key="img_upload"
    )

    if uploaded_img:
        img_pil   = Image.open(uploaded_img).convert("RGB")
        img_np    = np.array(img_pil)
        img_bgr   = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        annotated_bgr, detections = detect_animals(model, img_bgr, conf=conf_threshold)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        highest = get_highest_danger(detections)

        col_img, col_info = st.columns([2, 1])

        with col_img:
            st.image(annotated_rgb, caption="Detection Result", use_column_width=True)

        with col_info:
            st.markdown("### 🚨 Alert Status")
            if highest == "HIGH":
                st.markdown(
                    '<div class="alert-HIGH">🔴 DANGER — High-risk animal detected!</div>',
                    unsafe_allow_html=True,
                )
            elif highest == "MEDIUM":
                st.markdown(
                    '<div class="alert-MEDIUM">🟡 CAUTION — Medium-risk animal nearby</div>',
                    unsafe_allow_html=True,
                )
            elif highest == "LOW":
                st.markdown(
                    '<div class="alert-LOW">🟢 SAFE — Low-risk animal only</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alert-NONE">⚫ No animals detected</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### 📊 Detections")
            if detections:
                for d in detections:
                    emoji = DANGER_EMOJI[d["danger"]]
                    st.markdown(
                        f"{emoji} **{d['label'].title()}** — "
                        f"{d['danger']} danger | conf: `{d['confidence']}`"
                    )
            else:
                st.info("No animals found in this image.")
    else:
        st.info("⬆️ Upload an image to start detection.")


# ----------------------------------------------------------
# TAB 2 — VIDEO
# ----------------------------------------------------------
with tab2:
    st.subheader("Upload a Video")
    uploaded_vid = st.file_uploader(
        "Upload video (mp4 / avi / mov)", type=["mp4", "avi", "mov"], key="vid_upload"
    )

    if uploaded_vid:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap      = cv2.VideoCapture(tmp_path)
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_vid  = cap.get(cv2.CAP_PROP_FPS) or 25
        skip     = max(1, int(fps_vid // 5))   # process ~5 FPS regardless of source FPS

        st.info(f"Video loaded — {total} frames at {fps_vid:.1f} FPS. Processing every {skip} frame(s).")

        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar       = st.progress(0)
        stop_btn           = st.button("⏹ Stop Processing")

        frame_idx = 0
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress_bar.progress(min(frame_idx / max(total, 1), 1.0))

            if frame_idx % skip != 0:
                continue

            annotated_bgr, detections = detect_animals(model, frame, conf=conf_threshold)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            highest       = get_highest_danger(detections)

            frame_placeholder.image(annotated_rgb, use_column_width=True)

            if highest == "HIGH":
                status_placeholder.error("🔴 DANGER — High-risk animal detected!")
            elif highest == "MEDIUM":
                status_placeholder.warning("🟡 CAUTION — Medium-risk animal nearby")
            elif highest == "LOW":
                status_placeholder.success("🟢 SAFE — Low-risk animal only")

        cap.release()
        os.unlink(tmp_path)
        progress_bar.progress(1.0)
        st.success("✅ Video processing complete!")
    else:
        st.info("⬆️ Upload a video to start detection.")


# ----------------------------------------------------------
# TAB 3 — WEBCAM (Live)
# ----------------------------------------------------------
with tab3:
    st.subheader("Live Webcam Detection")
    st.warning(
        "⚠️ Webcam requires local setup. "
        "If running on Streamlit Cloud, use Image or Video tab instead."
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        start_btn = st.button("▶ Start Webcam", type="primary")
    with col_ctrl2:
        stop_cam  = st.button("⏹ Stop Webcam")

    frame_ph  = st.empty()
    alert_ph  = st.empty()
    stats_ph  = st.empty()

    if start_btn:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Check if it's connected and not used by another app.")
        else:
            high_count = med_count = low_count = frame_count = 0

            while not stop_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Lost webcam feed.")
                    break

                frame_count += 1
                annotated_bgr, detections = detect_animals(model, frame, conf=conf_threshold)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                highest       = get_highest_danger(detections)

                # Tally
                if highest == "HIGH":   high_count += 1
                elif highest == "MEDIUM": med_count += 1
                elif highest == "LOW":    low_count += 1

                frame_ph.image(annotated_rgb, channels="RGB", use_column_width=True)

                if highest == "HIGH":
                    alert_ph.error("🔴 DANGER — High-risk animal detected!")
                elif highest == "MEDIUM":
                    alert_ph.warning("🟡 CAUTION — Medium-risk animal nearby")
                elif highest == "LOW":
                    alert_ph.success("🟢 SAFE — Low-risk animal only")
                else:
                    alert_ph.info("👁️ Monitoring... No animals detected")

                stats_ph.markdown(
                    f"**Frames:** {frame_count} | "
                    f"🔴 HIGH: {high_count} | "
                    f"🟡 MEDIUM: {med_count} | "
                    f"🟢 LOW: {low_count}"
                )

                time.sleep(0.05)   # ~20 FPS cap

            cap.release()
            st.success("✅ Webcam stopped.")