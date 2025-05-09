import os
import sys
import tempfile
import cv2
import streamlit as st

from settings import get_barbell_curl, get_bent_over_dumbbell_row, get_squat_with_weights
from activity import Activity
from utils import get_mediapipe_pose

# --------------------- Custom CSS for Full-Width Upload and White Text ---------------------
def set_background_color():
    st.markdown(
        """
        <style>
        /* Set the background color */
        body {
            background-color: #f0f2f6;  /* Light gray background */
        }

        /* Style all text to be white */
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: white !important;
        }

        /* Style headers */
        h1 {
            font-weight: bold;
        }

        /* Style buttons */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }

        /* Style the upload section to take full width */
        .stFileUploader {
            width: 100% !important; /* Ensure upload spans full width */
        }

        /* Style the upload container */
        .upload-container {
            width: 100%; /* Full-width upload container */
            margin: auto; /* Center-align */
        }

        /* Add padding for sections */
        section.main > div {
            padding: 1rem;
        }

        /* Style the upload warning */
        .warning {
            color: red !important;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_color()

# --------------------- App Initialization ---------------------
# Define the base directory and append it to the system path
BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

# Set the title of the Streamlit application
st.title('🏋️‍♂️ Weight Training Analysis')

# --------------------- Sidebar for Activity Selection and Video Preview ---------------------
st.sidebar.header('🛠️ Configuration')

# Radio button for selecting the desired exercise activity
activity = st.sidebar.radio(
    'Select Activity',
    ['Barbell Curl', 'Bent Over Dumbbell Row', 'Squat with Weights'],
    horizontal=True
)

# Fetch settings based on the selected activity
if activity == 'Barbell Curl':
    settings = get_barbell_curl()
elif activity == 'Bent Over Dumbbell Row':
    settings = get_bent_over_dumbbell_row()
elif activity == 'Squat with Weights':
    settings = get_squat_with_weights()

# Initialize the Activity processor and Mediapipe pose detector
upload = Activity(settings=settings, flip_frame=True)
pose = get_mediapipe_pose()

# Initialize download flag in session state
if 'download' not in st.session_state:
    st.session_state['download'] = False

# Define the output video file path
output_video_file = 'output_recorded.mp4'

# Remove existing output video file if it exists
if os.path.exists(output_video_file):
    os.remove(output_video_file)

# --------------------- Sidebar for Video Preview ---------------------
st.sidebar.header('🎥 Input Video Preview')
input_video_placeholder = st.sidebar.empty()
warning_placeholder = st.sidebar.empty()

# --------------------- Main Content Area ---------------------
# Create a full-width upload section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

st.header('📤 Upload Your Video')  # Header
with st.form('Upload', clear_on_submit=True):
    up_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])  # Full-width uploader
    uploaded = st.form_submit_button("Upload")  # Upload button

st.markdown('</div>', unsafe_allow_html=True)

# Create a placeholder for the processed video frames
stframe = st.empty()

# Placeholder for download button
download_button_placeholder = st.empty()

if up_file and uploaded:
    download_button_placeholder.empty()
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    try:
        # Write the uploaded file to a temporary file
        temp_file.write(up_file.read())

        # Open the video file using OpenCV
        video_capture = cv2.VideoCapture(temp_file.name)

        # Retrieve video properties
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)

        # Define the codec and initialize the VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

        # Display the input video in the sidebar
        input_video_placeholder.markdown(
            '<p style="font-family:Helvetica; font-weight: bold; font-size: 16px;">Input Video</p>',
            unsafe_allow_html=True
        )
        input_video_placeholder.video(temp_file.name)

        # Process each frame of the video
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame based on the selected activity
            if activity == 'Barbell Curl':
                processed_frame, _ = upload.process_barbell_curl(rgb_frame, pose)
            elif activity == 'Bent Over Dumbbell Row':
                processed_frame, _ = upload.process_bent_over_dumbbell_row(rgb_frame, pose)
            elif activity == 'Squat with Weights':
                processed_frame, _ = upload.process_squat_with_weights(rgb_frame, pose)

            # Display the processed frame
            stframe.image(processed_frame, channels="RGB")

            # Write the processed frame to the output video
            video_writer.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

        # Release video resources
        video_capture.release()
        video_writer.release()
        stframe.empty()
        temp_file.close()

    except AttributeError:
        # Display a warning if processing fails
        warning_placeholder.markdown(
            '<p class="warning">Please upload a valid video file!</p>',
            unsafe_allow_html=True
        )

# If the output video exists, provide a download button
if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as video_file:
        download = download_button_placeholder.download_button(
            '⬇️ Download Processed Video',
            data=video_file,
            file_name='output_recorded.mp4'
        )

    # Update session state upon download
    if download:
        st.session_state['download'] = True

# Clean up the output video file after download
if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button_placeholder.empty()
