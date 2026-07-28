"""
Simple Helmet Detection App (Streamlit + YOLOv8)
--------------------------------------------------
Detects helmets in an uploaded image, an uploaded video, or a camera snapshot.

Run with:
    streamlit run app.py

Requirements (see requirements.txt):
    pip install streamlit ultralytics opencv-python-headless pillow numpy huggingface_hub
"""

import base64
import torch

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Helmet Detection", page_icon="⛑️", layout="centered")
st.title("⛑️ Helmet Detection")
st.write("Upload an image or video, or take a snapshot, to detect helmets.")

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Settings")

model = st.sidebar.radio(
    "Model",
    ["Pretrained hard-hat model (auto-download)", "Custom model path"],
)

HARD_HAT_MODELS = {
    "Nano (fastest)": "keremberke/yolov8n-hard-hat-detection",
    "Small (balanced)": "keremberke/yolov8s-hard-hat-detection",
    "Medium (most accurate)": "keremberke/yolov8m-hard-hat-detection",
}

if model== "Pretrained hard-hat model (auto-download)":
    size_choice = st.sidebar.selectbox("Model size", list(HARD_HAT_MODELS.keys()))
    repo_id = HARD_HAT_MODELS[size_choice]
    model_path = None  # resolved below by get_model()
else:
    repo_id = None
    model_path = st.sidebar.text_input(
        "Model path (.pt file)",
        value="best.pt",
        help=(
            "Path to your own YOLOv8 model trained to detect helmets. "
            "A generic yolov8n.pt (COCO) will NOT work — it has no helmet class."
        ),
    )

conf_thresh = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)

# Alarm sound toggle - plays "alarm.mp3" whenever a person without a helmet is detected
alarm_enabled = st.sidebar.toggle("🔊 Play alarm on violation", value=True)

source = st.sidebar.radio("Input source", ["Image", "Video", "Camera", "Live Webcam"])


# ----------------------------------------------------------------------
# Load model (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_from_path(path):
    return YOLO(path)


@st.cache_resource
def load_model_from_hub(repo):
    weights_path = hf_hub_download(repo_id=repo, filename="best.pt")
    return YOLO(weights_path)


def get_model():
    try:
        if repo_id is not None:
            with st.spinner(f"Downloading model ({repo_id})..."):
                return load_model_from_hub(repo_id)
        return load_model_from_path(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()


# ----------------------------------------------------------------------
# Alarm sound helpers
# ----------------------------------------------------------------------
@st.cache_resource
def load_alarm_audio_b64():
    """Read alarm.mp3 once and cache its base64 encoding for embedding in HTML <audio> tags."""
    try:
        with open("alarm.mp3", "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        st.sidebar.warning(f"Could not load alarm.mp3: {e}")
        return None


def play_alarm(placeholder):
    """Play alarm.mp3 once via an autoplaying, hidden HTML5 audio element rendered into `placeholder`."""
    audio_b64 = load_alarm_audio_b64()
    if not audio_b64:
        return
    placeholder.markdown(
        f"""
        <audio autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )


def annotate_frame(frame_bgr, model, conf):
    """Run detection on a single BGR frame.
    Returns:
        annotated: BGR frame with boxes drawn
        helmet_detected: True if at least one helmet/hardhat is detected
        total_people: Count of people in the frame
        no_helmet_count: Count of people not wearing a helmet
    """
    # results = model.predict(frame_bgr, conf=conf, verbose=False)
    device = 0 if torch.cuda.is_available() else "cpu"

    results = model.predict(
        frame_bgr,
        imgsz=640,
        conf=conf,
        verbose=False,
        half=torch.cuda.is_available(),
        device=device,
        max_det=25
    )   
    # annotated = results[0].plot()  # returns BGR numpy array with boxes drawn
    annotated = results[0].plot()
    
    helmet_detected = False
    total_people = 0
    no_helmet_count = 0
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names
        
        helmet_count = 0
        no_helmet_count_local = 0
        person_count = 0
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = names.get(cls_id, "").lower()
            
            # Exclude negative matches (e.g. no-hardhat, no helmet)
            is_negative = any(neg in class_name for neg in ["no-", "no_", "no ", "without", "bare", "not"])
            is_positive = any(pos in class_name for pos in ["hardhat", "helmet", "safety_helmet"])
            is_person = any(p in class_name for p in ["person", "worker", "human", "man", "woman"])
            is_head = "head" in class_name
            
            if is_positive and not is_negative:
                helmet_count += 1
                helmet_detected = True
            elif is_negative or (is_head and not is_positive):
                no_helmet_count_local += 1
            elif is_person:
                person_count += 1
                
            # Fallback: if there's no explicit positive match but the class name is not negative and the dataset is a 1-class dataset
            if not is_positive and not is_negative and not is_person and not is_head and len(names) == 1:
                helmet_count += 1
                helmet_detected = True

        # Compute counts based on classes present in the model
        if helmet_count > 0 or no_helmet_count_local > 0:
            total_people = helmet_count + no_helmet_count_local
            no_helmet_count = no_helmet_count_local
            if person_count > total_people:
                total_people = person_count
                no_helmet_count = max(no_helmet_count_local, person_count - helmet_count)
        elif person_count > 0:
            total_people = person_count
            no_helmet_count = person_count  # since helmet_count is 0
        else:
            total_people = 0
            no_helmet_count = 0
            
    return annotated, helmet_detected, total_people, no_helmet_count


def display_stats(total_people, no_helmet_count):
    """Render a premium dark-themed statistics dashboard component."""
    wearing_helmet = max(0, total_people - no_helmet_count)
    st.markdown(
        f"""
        <div style="
            background: #1e1e24;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 1px solid #2e2e38;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        ">
            <h3 style="color: #f1f1f6; margin-top: 0; margin-bottom: 15px; font-size: 1.2rem; font-weight: 600; text-align: center;">📊 Frame Analysis Statistics</h3>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px;">
                <div style="flex: 1; min-width: 100px; text-align: center; background: #2a2a35; padding: 12px; border-radius: 8px; border: 1px solid #38bdf8;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8;">{total_people}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Total People</div>
                </div>
                <div style="flex: 1; min-width: 100px; text-align: center; background: #2a2a35; padding: 12px; border-radius: 8px; border: 1px solid #34d399;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #34d399;">{wearing_helmet}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Wearing Helmet</div>
                </div>
                <div style="flex: 1; min-width: 100px; text-align: center; background: #2a2a35; padding: 12px; border-radius: 8px; border: 1px solid { '#ef4444' if no_helmet_count > 0 else '#2e2e38' };">
                    <div style="font-size: 1.8rem; font-weight: 700; color: { '#ef4444' if no_helmet_count > 0 else '#94a3b8' };">{no_helmet_count}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Not Wearing Helmet</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def trigger_no_helmet_popup(total_people, no_helmet_count):
    """Trigger a premium non-blocking floating warning card about the safety violation."""
    st.markdown(
        f"""
        <div id="helmet-popup" style="
            position: fixed;
            top: 70px;
            right: 20px;
            width: 320px;
            background: linear-gradient(135deg, #1e1e24, #141417);
            color: #f1f1f6;
            padding: 18px;
            border-radius: 12px;
            border-left: 5px solid #ef4444;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            z-index: 999999;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        ">
            <div style="display: flex; gap: 12px; align-items: flex-start;">
                <div style="font-size: 26px; line-height: 1; animation: pulse-alert 2s infinite;">🚨</div>
                <div style="flex-grow: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #ef4444; font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Safety Violation</span>
                        <button onclick="
                            window.helmetPopupDismissedUntil = Date.now() + 3000;
                            document.getElementById('helmet-popup').style.display='none';
                        " style="
                            background: none;
                            border: none;
                            color: #94a3b8;
                            font-size: 16px;
                            cursor: pointer;
                            padding: 0;
                            line-height: 1;
                        "></button>
                    </div>
                    <p style="font-size: 13px; color: #cbd5e1; margin: 8px 0 12px 0; line-height: 1.4;">
                        Personnel without safety helmets detected.
                    </p>
                    <div style="display: flex; gap: 10px; background: #25252e; padding: 8px 12px; border-radius: 6px; margin-bottom: 12px;">
                        <div style="flex: 1; text-align: center;">
                            <div style="font-size: 14px; font-weight: bold; color: #38bdf8;">{total_people}</div>
                            <div style="font-size: 10px; color: #94a3b8;">Total People</div>
                        </div>
                        <div style="border-left: 1px solid #3f3f46; height: 20px; align-self: center;"></div>
                        <div style="flex: 1; text-align: center;">
                            <div style="font-size: 14px; font-weight: bold; color: #ef4444;">{no_helmet_count}</div>
                            <div style="font-size: 10px; color: #94a3b8;">Not Wearing</div>
                        </div>
                    </div>
                    <button onclick="
                        window.helmetPopupDismissedUntil = Date.now() + 3000;
                        document.getElementById('helmet-popup').style.display='none';
                    " style="
                        background: #ef4444;
                        border: none;
                        color: white;
                        padding: 6px 12px;
                        font-size: 11px;
                        font-weight: 600;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: background 0.2s;
                        width: 100%;
                    ">
                        Dismiss Alert
                    </button>
                </div>
            </div>
        </div>
        <script>
            (function() {{
                try {{
                    const now = Date.now();
                    if (window.helmetPopupDismissedUntil && window.helmetPopupDismissedUntil > now) {{
                        document.getElementById('helmet-popup').style.display = 'none';
                    }}
                }} catch(e) {{
                    console.error("Error checking dismiss state:", e);
                }}
            }})();
        </script>
        <style>
            @keyframes slideIn {{
                from {{ transform: translateX(120%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            @keyframes pulse-alert {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.1); }}
                100% {{ transform: scale(1); }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ----------------------------------------------------------------------
# IMAGE MODE
# ----------------------------------------------------------------------
if source == "Image":
    uploaded_img = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_img is not None:
        model = get_model()

        image = Image.open(uploaded_img).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting helmets..."):
            annotated_bgr, helmet_detected, total_people, no_helmet_count = annotate_frame(frame_bgr, model, conf_thresh)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection result", use_container_width=True)
        
        # Display analysis statistics
        display_stats(total_people, no_helmet_count)

        alarm_holder = st.empty()

        # Trigger popup if no helmet is detected
        if no_helmet_count > 0:
            trigger_no_helmet_popup(total_people, no_helmet_count)
            if alarm_enabled:
                play_alarm(alarm_holder)
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #ff416c, #ff4b2b);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 1.1rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin: 10px 0;
                ">
                    ❌ Personnel Without Helmet Detected
                </div>
                """,
                unsafe_allow_html=True
            )
        elif total_people > 0:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #11998e, #38ef7d);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 1.1rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin: 10px 0;
                ">
                    ✅ All Personnel Wearing Helmets
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #718096, #4a5568);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 1.1rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin: 10px 0;
                ">
                    ℹ️ No Personnel Detected in Frame
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------------------------------------------------------
# VIDEO MODE
# ----------------------------------------------------------------------
elif source == "Video":
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        model = get_model()

        # Save upload to a temp file so OpenCV can read it
        temp_in_path = "temp_input_video.mp4"
        with open(temp_in_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stframe = st.empty()
        status_holder = st.empty()
        stats_holder = st.empty()
        popup_holder = st.empty()
        alarm_holder = st.empty()
        progress = st.progress(0)
        frame_count = 0

        run_detection = st.button("Start Detection")

        if run_detection:
            popup_triggered = False
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_bgr, helmet_detected, total_people, no_helmet_count = annotate_frame(frame, model, conf_thresh)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

                # Update statistics dashboard dynamically
                with stats_holder:
                    display_stats(total_people, no_helmet_count)

                # Trigger warning popup when a violation is detected (JS handling dismissal/cooldown)
                if no_helmet_count > 0:
                    with popup_holder:
                        trigger_no_helmet_popup(total_people, no_helmet_count)
                    if alarm_enabled:
                        play_alarm(alarm_holder)
                else:
                    popup_holder.empty()
                    alarm_holder.empty()
                    popup_triggered = False

                # Update status badge dynamically
                if no_helmet_count > 0:
                    status_holder.markdown(
                        """
                        <div style="
                            background: linear-gradient(135deg, #ff416c, #ff4b2b);
                            color: white;
                            padding: 12px 20px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: bold;
                            font-size: 1.1rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            ❌ Personnel Without Helmet Detected
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif total_people > 0:
                    status_holder.markdown(
                        """
                        <div style="
                            background: linear-gradient(135deg, #11998e, #38ef7d);
                            color: white;
                            padding: 12px 20px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: bold;
                            font-size: 1.1rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            ✅ All Personnel Wearing Helmets
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    status_holder.markdown(
                        """
                        <div style="
                            background: linear-gradient(135deg, #718096, #4a5568);
                            color: white;
                            padding: 12px 20px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: bold;
                            font-size: 1.1rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            ℹ️ No Personnel Detected in Frame
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                frame_count += 1
                if total_frames:
                    progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            st.success("Video processing complete.")

# ----------------------------------------------------------------------
# CAMERA MODE (snapshot based - simplest option, no extra deps needed)
# ----------------------------------------------------------------------
elif source == "Camera":
    camera_img = st.camera_input("Take a picture")

    if camera_img is not None:
        model = get_model()

        image = Image.open(camera_img).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting helmets..."):
            annotated_bgr, helmet_detected, total_people, no_helmet_count = annotate_frame(frame_bgr, model, conf_thresh)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection result", use_container_width=True)

        # Display analysis statistics
        display_stats(total_people, no_helmet_count)

        alarm_holder = st.empty()

        # Trigger warning popup
        if no_helmet_count > 0:
            trigger_no_helmet_popup(total_people, no_helmet_count)
            if alarm_enabled:
                play_alarm(alarm_holder)
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #ff416c, #ff4b2b);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 1.1rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin: 10px 0;
                ">
                    ❌ Personnel Without Helmet Detected
                </div>
                """,
                unsafe_allow_html=True
            )
        elif total_people > 0:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #11998e, #38ef7d);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 1.1rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin: 10px 0;
                ">
                    ✅ All Personnel Wearing Helmets
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #718096, #4a5568);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 1.1rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin: 10px 0;
                ">
                    ℹ️ No Personnel Detected in Frame
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------------------------------------------------------
# LIVE WEBCAM MODE (continuous real-time detection)
# ----------------------------------------------------------------------
elif source == "Live Webcam":

    def scan_local_cameras(max_index=8):
        """Probe camera indices 0..max_index-1 and return the ones that open successfully."""
        available = []
        for idx in range(max_index):
            cap_test = cv2.VideoCapture(idx)
            if cap_test is not None and cap_test.isOpened():
                # Try to grab a frame to make sure it's a real, usable device
                ok, _ = cap_test.read()
                if ok:
                    available.append(idx)
            cap_test.release()
        return available

    st.subheader("Camera source")

    video_source = None  # what gets passed to cv2.VideoCapture()

    col_local, col_ip = st.columns(2)

    # ---- Left column: local webcam ----
    with col_local:
        st.markdown("**🎥 Local webcam**")
        if st.button("🔍 Scan for cameras", use_container_width=True):
            with st.spinner("Checking connected cameras..."):
                st.session_state["available_cameras"] = scan_local_cameras()

        available_cameras = st.session_state.get("available_cameras")

        if available_cameras is None:
            st.caption("Click scan to detect webcams on this machine.")
        elif len(available_cameras) == 0:
            st.caption("⚠️ No local cameras detected.")
        else:
            labels = {idx: f"Camera {idx}" for idx in available_cameras}
            chosen_idx = st.selectbox(
                "Select a camera",
                options=available_cameras,
                format_func=lambda i: labels[i],
                key="chosen_camera_index",
                label_visibility="collapsed",
            )
            local_source = chosen_idx
        if available_cameras:
            use_local = st.checkbox("Use local webcam", key="use_local_cam")
            if use_local:
                video_source = local_source

    # ---- Right column: IP camera ----
    with col_ip:
        st.markdown("**🌐 IP camera**")
        SAVED_IP_CAMERAS = {
            "Site camera (channel 1)": "rtsp://admin:Ntpc%40123@192.168.1.250:554/video/live?channel=1&subtype=1",
        }
        ip_choice = st.selectbox(
            "IP camera",
            options=list(SAVED_IP_CAMERAS.keys()) + ["Enter your own URL"],
            key="ip_camera_choice",
            label_visibility="collapsed",
        )

        if ip_choice == "Enter your own URL":
            ip_url = st.text_input(
                "IP camera URL",
                value="",
                placeholder="rtsp://user:pass@192.168.1.10:554/stream1",
                label_visibility="collapsed",
            )
        else:
            ip_url = SAVED_IP_CAMERAS[ip_choice]
            st.caption(f"`{ip_url}`")

        btn_col, use_col = st.columns(2)
        with btn_col:
            test_clicked = st.button("Test", use_container_width=True)
        with use_col:
            use_ip = st.checkbox("Use this", key="use_ip_cam")

        if test_clicked and ip_url:
            with st.spinner("Connecting..."):
                test_cap = cv2.VideoCapture(ip_url)
                ok = test_cap.isOpened()
                if ok:
                    ok, _ = test_cap.read()
                test_cap.release()
            if ok:
                st.success("Connected ✅", icon="✅")
            else:
                st.error("Connection failed. Check URL/credentials/network.")

        if use_ip and ip_url:
            video_source = ip_url

    run_live = st.toggle("Start Live Webcam Feed", value=False, disabled=(video_source is None))
    if video_source is None:
        st.caption("Pick a camera on the left or right (and check its 'Use' box) to enable the live feed.")

    if run_live and video_source is not None:
        model = get_model()
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        # Create empty placeholders
        stframe = st.empty()
        status_holder = st.empty()
        stats_holder = st.empty()
        popup_holder = st.empty()
        alarm_holder = st.empty()
        
        try:
            popup_triggered = False
            while run_live:
                # ret, frame = cap.read()
                for _ in range(3):
                    cap.grab()

                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access the camera. Please make sure it's connected/reachable and not in use by another app.")
                    break
                    
                annotated_bgr, helmet_detected, total_people, no_helmet_count = annotate_frame(frame, model, conf_thresh)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
                stframe.image(annotated_rgb, channels="RGB", use_container_width=True)
                
                # Update statistics dashboard dynamically
                with stats_holder:
                    display_stats(total_people, no_helmet_count)

                # Trigger warning popup when a violation is detected (JS handling dismissal/cooldown)
                if no_helmet_count > 0:
                    with popup_holder:
                        trigger_no_helmet_popup(total_people, no_helmet_count)
                    
                    if alarm_enabled:
                        play_alarm(alarm_holder)
                else:
                    popup_holder.empty()
                    alarm_holder.empty()
                    popup_triggered = False

                # Update status badge dynamically
                if no_helmet_count > 0:
                    status_holder.markdown(
                        """
                        <div style="
                            background: linear-gradient(135deg, #ff416c, #ff4b2b);
                            color: white;
                            padding: 12px 20px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: bold;
                            font-size: 1.1rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            ❌ Personnel Without Helmet Detected
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif total_people > 0:
                    status_holder.markdown(
                        """
                        <div style="
                            background: linear-gradient(135deg, #11998e, #38ef7d);
                            color: white;
                            padding: 12px 20px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: bold;
                            font-size: 1.1rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            ✅ All Personnel Wearing Helmets
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    status_holder.markdown(
                        """
                        <div style="
                            background: linear-gradient(135deg, #718096, #4a5568);
                            color: white;
                            padding: 12px 20px;
                            border-radius: 8px;
                            text-align: center;
                            font-weight: bold;
                            font-size: 1.1rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin: 10px 0;
                        ">
                            ℹ️ No Personnel Detected in Frame
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        finally:
            cap.release()

st.sidebar.markdown("---")
st.sidebar.caption(
    "The pretrained model detects two classes: 'Hardhat' and 'NO-Hardhat'. "
    "The first run will download the weights (a few MB to ~50MB depending on size) "
    "and cache them for later runs."
)