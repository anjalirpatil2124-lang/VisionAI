import streamlit as st
from database import (
    save_detection,
    get_detection_history,
    get_detection_details,
    get_database_analytics,
    create_user,
    authenticate_user
)

import cv2
import tempfile
import os
from PIL import Image
from collections import Counter

from analytics import record_detection
import pandas as pd
import plotly.express as px

from detection import detect_image
from video_detection import detect_video
from streamlit_webrtc import webrtc_streamer
from webcam_detection import WebcamProcessor
from report_generator import generate_detection_report


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VisionAI | Smart Object Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #F5F8FF;
    color: #1E293B;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Headings */

h1, h2, h3 {
    color: #1E3A5F;
    font-weight: 700;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] h1 {
    color: #244B8A;
}

/* Buttons */

div.stButton > button {
    background-color: #4169E1;
    color: white;
    border-radius: 9px;
    border: none;
    height: 45px;
    font-weight: 600;
    transition: 0.2s;
}

div.stButton > button:hover {
    background-color: #294FC1;
    color: white;
    border: none;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 12px rgba(30, 58, 95, 0.05);
}

/* Feature cards */

.feature-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 15px;
    padding: 25px;
    min-height: 145px;
    box-shadow: 0 3px 12px rgba(30, 58, 95, 0.04);
}

.feature-icon {
    font-size: 30px;
}

.feature-title {
    color: #244B8A;
    font-size: 19px;
    font-weight: 700;
    margin-top: 12px;
}

.feature-description {
    color: #64748B;
    font-size: 14px;
}

/* Dividers */

hr {
    border-color: #E2E8F0;
}

/* Dataframes */

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid #E2E8F0;
}

/* Expanders */

[data-testid="stExpander"] {
    background-color: white;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
}

/* Responsive design */

@media (max-width: 768px) {

    .block-container {
        padding: 1rem;
    }

    h1 {
        font-size: 27px !important;
    }

    div[data-testid="stMetric"] {
        padding: 12px;
    }

    .feature-card {
        margin-bottom: 12px;
    }

}

</style>
""", unsafe_allow_html=True)


# ==========================================
# USER AUTHENTICATION
# ==========================================

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.user_id is None:
    st.markdown("<h1 style='text-align:center;'>👁️ VisionAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748B;'>Sign in or create an account to use your personal object detection workspace.</p>", unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.5, 1])
    with center:
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username", key="login_username")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submitted = st.form_submit_button("Login", use_container_width=True)
            if login_submitted:
                user = authenticate_user(login_username, login_password)
                if user:
                    st.session_state.user_id = user["id"]
                    st.session_state.username = user["username"]
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input("Choose a username", key="register_username")
                new_password = st.text_input("Choose a password", type="password", key="register_password")
                confirm_password = st.text_input("Confirm password", type="password", key="confirm_password")
                register_submitted = st.form_submit_button("Create Account", use_container_width=True)
            if register_submitted:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    created, message = create_user(new_username, new_password)
                    if created:
                        st.success(message + " Please log in.")
                    else:
                        st.error(message)
    st.stop()



# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown(
    """
    <div style="padding:10px 0 20px 0;">
        <h1 style="color:#244B8A; font-size:29px;">
            👁️ VisionAI
        </h1>
        <p style="color:#64748B; font-size:14px;">
            Intelligent Computer Vision System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(f"**Signed in as:** {st.session_state.username}")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()

st.sidebar.divider()

st.sidebar.markdown("### NAVIGATION")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard",
        "Image Detection",
        "Video Detection",
        "Live Detection",
        "Analytics",
        "Detection History",
        "About Project"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:15px 0;">
        <p style="color:#64748B; font-size:12px;">
            VisionAI v1.0
        </p>
        <p style="color:#94A3B8; font-size:11px;">
            AI-Powered Object Detection
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown(
        """
        <h1 style="font-size:38px; margin-bottom:5px;">
            Welcome to VisionAI 👋
        </h1>

        <p style="color:#64748B; font-size:18px;">
            AI-Powered Object Detection and Analysis Platform
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Explore VisionAI")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🖼️</div>
                <div class="feature-title">Image Detection</div>
                <p class="feature-description">
                    Upload images and identify multiple objects
                    using YOLO AI.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎥</div>
                <div class="feature-title">Video Detection</div>
                <p class="feature-description">
                    Analyse video frames and identify
                    objects automatically.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📷</div>
                <div class="feature-title">Live Detection</div>
                <p class="feature-description">
                    Detect objects instantly using
                    your webcam.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Analytics</div>
                <p class="feature-description">
                    View detection statistics and
                    interactive charts.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🗂️</div>
                <div class="feature-title">Detection History</div>
                <p class="feature-description">
                    Access previously saved
                    detection sessions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col6:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">YOLO AI Engine</div>
                <p class="feature-description">
                    Intelligent object identification
                    powered by deep learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.divider()

    st.subheader("Get Started")

    st.info(
        "Select Image Detection, Video Detection or "
        "Live Detection from the sidebar to begin."
    )


# ==========================================
# IMAGE DETECTION
# ==========================================

elif page == "Image Detection":

    st.title("🖼️ Image Object Detection")

    st.caption(
        "Upload an image to identify objects using YOLO AI."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("🔍 Detect Objects"):

            with st.spinner("AI is analysing your image..."):

                temp_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False
                    ) as temp_file:

                        image.save(temp_file, format="JPEG")
                        temp_path = temp_file.name

                    result_image, objects = detect_image(temp_path)

                    record_detection("Image", objects)

                    save_detection("Image", objects, user_id=st.session_state.user_id)

                    result_rgb = cv2.cvtColor(
                        result_image,
                        cv2.COLOR_BGR2RGB
                    )

                    st.success("Detection completed successfully!")

                    st.subheader("Detection Results")

                    st.image(
                        result_rgb,
                        use_container_width=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Total Objects",
                            len(objects)
                        )

                    with col2:

                        avg_confidence = (
                            sum(o["confidence"] for o in objects)
                            / len(objects)
                            if objects else 0
                        )

                        st.metric(
                            "Average Confidence",
                            f"{avg_confidence:.2f}%"
                        )

                    if objects:

                        st.subheader("Detected Object Details")

                        counts = Counter(
                            obj["name"] for obj in objects
                        )

                        for name, count in counts.items():

                            st.write(
                                f"**{name.title()}**: {count}"
                            )

                        st.dataframe(
                            pd.DataFrame(objects),
                            use_container_width=True
                        )

                    else:

                        st.warning("No objects detected.")

                    success, encoded_image = cv2.imencode(
                        ".jpg",
                        result_rgb
                    )

                    if success:

                        st.download_button(
                            "⬇️ Download Detected Image",
                            data=encoded_image.tobytes(),
                            file_name="detected_image.jpg",
                            mime="image/jpeg"
                        )

                except Exception as e:

                    st.error(f"Detection failed: {e}")

                finally:

                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)


# ==========================================
# VIDEO DETECTION
# ==========================================

elif page == "Video Detection":

    st.title("🎥 Video Object Detection")

    st.caption(
        "Upload a video and identify objects frame by frame."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.subheader("Original Video")

        st.video(uploaded_video)

        if st.button("▶️ Start Video Detection"):

            os.makedirs("outputs", exist_ok=True)

            input_path = None
            output_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(
                        uploaded_video.name
                    )[1],
                    delete=False
                ) as temp_input:

                    temp_input.write(
                        uploaded_video.getbuffer()
                    )

                    input_path = temp_input.name

                output_path = os.path.join(
                    "outputs",
                    "detected_video.mp4"
                )

                with st.spinner(
                    "AI is processing your video..."
                ):

                    results = detect_video(
                        input_path,
                        output_path,
                        user_id=st.session_state.user_id
                    )

                st.success("Video detection completed!")

                st.subheader("Processed Video")

                st.video(output_path)

                st.metric(
                    "Frames Processed",
                    results["frames_processed"]
                )

                st.subheader("Detection Statistics")

                for name, count in results[
                    "object_counts"
                ].items():

                    st.write(
                        f"**{name.title()}**: {count}"
                    )

                with open(output_path, "rb") as video_file:

                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=video_file,
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:

                st.error(
                    f"Video processing failed: {e}"
                )

            finally:

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)


# ==========================================
# LIVE DETECTION
# ==========================================

elif page == "Live Detection":

    st.title("📷 Live Webcam Object Detection")

    st.caption(
        "Detect objects in real time using your camera."
    )

    st.info(
        "Allow camera permission when your browser asks."
    )

    ctx = webrtc_streamer(
        key="visionai-live",
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    if ctx.state.playing:

        st.success("Live detection is running!")

    else:

        st.warning(
            "Click START to activate your camera."
        )

    if st.button("💾 Save Webcam Results"):

        if ctx.video_processor is not None:

            results = ctx.video_processor.get_results()

            if results:

                save_detection("Webcam", results, user_id=st.session_state.user_id)

                st.success(
                    f"Successfully saved {len(results)} detections!"
                )

            else:

                st.warning("No objects detected yet.")

        else:

            st.warning("Start your webcam first.")


# ==========================================
# ANALYTICS DASHBOARD
# ==========================================

elif page == "Analytics":

    st.title("📊 VisionAI Analytics Dashboard")

    st.caption(
        "Complete overview of your AI object detection activities."
    )

    data = get_database_analytics(st.session_state.user_id)

    source_stats = {
        source: {
            "sessions": sessions,
            "objects": objects or 0
        }
        for source, sessions, objects in data["source_data"]
    }

    image_sessions = source_stats.get(
        "Image", {}
    ).get("sessions", 0)

    video_sessions = source_stats.get(
        "Video", {}
    ).get("sessions", 0)

    webcam_sessions = source_stats.get(
        "Webcam", {}
    ).get("sessions", 0)

    st.subheader("Overall Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Objects Detected",
        data["total_objects"]
    )

    col2.metric(
        "Detection Sessions",
        data["total_sessions"]
    )

    col3.metric(
        "Average Confidence",
        f'{data["average_confidence"]:.2f}%'
    )

    st.divider()

    st.subheader("Detection Sources")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Image Sessions",
        image_sessions
    )

    col2.metric(
        "Video Sessions",
        video_sessions
    )

    col3.metric(
        "Webcam Sessions",
        webcam_sessions
    )

    st.divider()

    st.subheader("Object Distribution")

    if data["object_data"]:

        object_df = pd.DataFrame(
            data["object_data"],
            columns=["Object", "Count"]
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                object_df,
                x="Object",
                y="Count",
                title="Object Detection Statistics",
                color="Object"
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#1E293B"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.pie(
                object_df,
                names="Object",
                values="Count",
                title="Object Distribution",
                hole=0.35
            )

            fig.update_layout(
                paper_bgcolor="white",
                font_color="#1E293B"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info("No object statistics available.")

    st.divider()

    st.subheader("Recent Detection Activity")

    history = get_detection_history(st.session_state.user_id)

    if history:

        history_df = pd.DataFrame(
            history,
            columns=[
                "ID",
                "Source",
                "Total Objects",
                "Average Confidence",
                "Date"
            ]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No detection history available.")


# ==========================================
# DETECTION HISTORY
# ==========================================

elif page == "Detection History":

    st.title("🗂️ Detection History")

    st.caption(
        "View all your previously saved detection sessions."
    )

    if st.button("🔄 Refresh History"):

        st.rerun()

    history = get_detection_history(st.session_state.user_id)

    if history:

        st.success(
            f"{len(history)} detection sessions found."
        )

        for record in history:

            run_id = record[0]
            source = record[1]
            total = record[2]
            confidence = record[3]
            date = record[4]

            with st.expander(
                f"{source} | {date} | {total} objects"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Total Objects",
                        total
                    )

                with col2:

                    st.metric(
                        "Average Confidence",
                        f"{confidence:.2f}%"
                    )

                details = get_detection_details(run_id, st.session_state.user_id)

                if details:

                    df = pd.DataFrame(
                        details,
                        columns=[
                            "Object",
                            "Confidence"
                        ]
                    )

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info("No object details available.")

                # Generate and download a PDF report for this session
                try:
                    pdf_report = generate_detection_report(record, details)

                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_report,
                        file_name=f"VisionAI_Report_Session_{run_id}.pdf",
                        mime="application/pdf",
                        key=f"download_pdf_{run_id}"
                    )

                except Exception as e:
                    st.error(f"Could not generate PDF report: {e}")

    else:

        st.info("No detection history available yet.")


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("👁️ About VisionAI")

    st.markdown(
        """
        ### Intelligent Computer Vision System

        VisionAI is an AI-powered computer vision application
        developed using Python, YOLO and OpenCV.

        The application identifies multiple objects in
        images, videos and live webcam streams.

        ### Key Features

        - Image Object Detection
        - Video Object Detection
        - Real-Time Webcam Detection
        - Object Confidence Scores
        - Detection Analytics
        - Individual User Accounts
        - Secure Login and Registration
        - Personal Detection History
        - Permanent Detection History

        ### Technologies Used

        - Python
        - YOLOv8
        - OpenCV
        - PyTorch
        - Streamlit
        - SQLite
        - Pandas
        - Plotly

        ### Project Objective

        To develop an intelligent object detection system
        that can identify objects, analyse visual data
        and provide useful detection statistics.
        """
    )

    st.divider()

    st.info(
        "VisionAI – Intelligent Object Detection System"
    )