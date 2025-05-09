import streamlit as st

# Apply custom CSS for advanced styling
st.markdown(
    """
    <style>
    /* Title Styling */
    .title {
        color: #4FBFFF; /* Light blue for the title */
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
        font-family: 'Helvetica', sans-serif;
    }

    /* Subtitle Styling */
    .subtitle {
        color: #A3D4FF; /* Softer light blue */
        font-size: 22px;
        font-style: italic;
        text-align: center;
        margin-top: -5px;
        font-family: 'Georgia', serif;
    }

    /* Details Section */
    .details {
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        font-size: 18px;
        text-align: center;
        color: #D1EFFF; /* Very light blue text for better contrast */
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }

    /* Abstract Section */
    .abstract {
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        font-size: 18px;
        text-align: left;
        color: #D1EFFF; /* Very light blue text for better contrast */
        font-family: 'Arial', sans-serif;
        line-height: 1.8;
    }

    /* Highlight Text */
    .highlight {
        color: #4FBFFF; /* Light blue for emphasis */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the application
st.markdown('<div class="title">Weight Training Analysis</div>', unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="subtitle">Introduction to Sports Engineering Project</div>', unsafe_allow_html=True)

# Information Section
st.markdown(
    """
    <div class="details">
        <p><span class="highlight">Name:</span> Muhammad Zaroon Hassan</p>
        <p><span class="highlight">Registration No:</span> 227309</p>
        <p>This project focuses on <strong>weight training analysis</strong> using Mediapipe for pose detection and real-time feedback. 
        It is designed to assist users in performing exercises with proper form and posture, enhancing both safety and performance.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# # Abstract Section
# st.markdown(
#     """
#     <div class="abstract">
#         <h3 style="color: #4FBFFF;">Abstract</h3>
#         <p>The system for analyzing the barbell curl exercise is built using real-time video processing with Mediapipe for pose estimation and OpenCV for visualization. 
#         Key aspects of the implementation include detecting body posture, calculating joint angles, and providing corrective feedback. The process integrates predefined settings, real-time landmark detection, and feedback mechanisms. Below is a summary of how this system works:</p>
#         <ul>
#             <li>Predefined thresholds and angle ranges for joint positions (e.g., wrist-shoulder-elbow angle) to identify correct and incorrect movements.</li>
#             <li>Mediapipe is used to extract body landmarks and compute key joint angles.</li>
#             <li>Real-time visual feedback (text overlays, highlights, and angle indicators) using OpenCV.</li>
#             <li>Tracking of exercise stages to validate repetitions and ensure proper form.</li>
#             <li>Audio and visual feedback for correcting errors like excessive swinging or improper shoulder alignment.</li>
#             <li>Monitoring inactivity and resetting counters after prolonged idle periods.</li>
#         </ul>
#         <p>This system provides an interactive way to monitor and improve exercise performance, ensuring proper form and minimizing the risk of injury.</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
