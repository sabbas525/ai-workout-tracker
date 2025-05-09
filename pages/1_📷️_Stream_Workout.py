# live_workout.py  ────────────────────────────────────────────────────
import datetime
import av, cv2, streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer, WebRtcMode

# ───── project-specific helpers ───────────────────────────────────────
from settings import (
    get_barbell_curl,
    get_bent_over_dumbbell_row,
    get_squat_with_weights,
)
from activity import Activity
from utils import get_mediapipe_pose


# --------------------- Custom CSS ---------------------
def set_custom_styles():
    st.markdown(
        """
        <style>
        body { background:#e6f7ff; }          /* keep your light theme */
        h1,h2,h3,h4,h5,h6,label,span,p { color:#fff!important; }

        /* force-hide spinner once frames arrive */
        .streamlit-webrtc-component .loading { display:none!important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


set_custom_styles()

# --------------------- UI -----------------------------
st.markdown(
    '<div style="color:red; font-size:16px;">Live workout feature is experimental</div>',
    unsafe_allow_html=True,
)
st.title("🏋️‍♂️ Weight Training Analysis")

st.sidebar.header("🔧 Configuration")
exercise_options = ["Barbell Curl", "Bent Over Dumbbell Row", "Squat with Weights"]
selected_activity = st.sidebar.radio("Select Activity", exercise_options, horizontal=True)

# switch lets you disable processing if you ever need raw video
do_processing = st.sidebar.checkbox("Enable Activity processing", value=True)

# --------------------- init helpers -------------------
settings_map = {
    "Barbell Curl": get_barbell_curl,
    "Bent Over Dumbbell Row": get_bent_over_dumbbell_row,
    "Squat with Weights": get_squat_with_weights,
}
activity_processor = Activity(settings=settings_map[selected_activity](), flip_frame=True)
pose_detector = get_mediapipe_pose()

st.markdown("---")
st.subheader("🎥 Live Video Feed")

# --------------------- frame callback -----------------
log_box = st.empty()  # first error shows up here


def process_frame_safe(frame: av.VideoFrame) -> av.VideoFrame:
    """
    • Always returns a frame so WebRTC keeps playing.
    • Converts colour spaces correctly.
    • Logs the first error of every type.
    """
    try:
        img_rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)

        if do_processing:
            if selected_activity == "Barbell Curl":
                out_rgb, _ = activity_processor.process_barbell_curl(img_rgb, pose_detector)
            elif selected_activity == "Bent Over Dumbbell Row":
                out_rgb, _ = activity_processor.process_bent_over_dumbbell_row(img_rgb, pose_detector)
            else:
                out_rgb, _ = activity_processor.process_squat_with_weights(img_rgb, pose_detector)

            # if Activity returns None or wrong shape → fall back
            if out_rgb is None or out_rgb.shape != img_rgb.shape:
                out_rgb = img_rgb
        else:
            out_rgb = img_rgb

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")

    except Exception as e:
        key = f"err:{type(e).__name__}"
        if key not in st.session_state:
            st.session_state[key] = True
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            log_box.error(f"{ts} – {repr(e)}")
            print("Frame-processing error:", repr(e))
        return frame  # send raw frame so stream stays alive


# --------------------- WebRTC streamer ----------------
webrtc_streamer(
    key="ai-weight-training-coach",
    mode=WebRtcMode.SENDRECV,
    async_processing=True,  # run callback in background thread
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=process_frame_safe,
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, width=720, muted=True),
)
