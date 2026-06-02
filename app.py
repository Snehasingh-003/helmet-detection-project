
# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import tempfile

# # -------------------------------
# # Page Configuration
# # -------------------------------
# st.set_page_config(
#     page_title="Helmet Detection System",
#     page_icon="🪖",
#     layout="centered"
# )

# st.title("🪖 Helmet Detection System")
# st.write("Upload an image to detect whether a helmet is present.")

# # -------------------------------
# # Load Model
# # -------------------------------
# MODEL_PATH = "best.pt"

# @st.cache_resource
# def load_model():
#     return YOLO(MODEL_PATH)

# try:
#     model = load_model()
# except Exception as e:
#     st.error(f"Model loading failed: {e}")
#     st.stop()


# # -------------------------------
# # Upload Image
# # -------------------------------
# uploaded_file = st.file_uploader(
#     "Choose an image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:

#     image = Image.open(uploaded_file).convert("RGB")

#     st.image(
#         image,
#         caption="Uploaded Image",
#         use_container_width=True
#     )

#     if st.button("Detect Helmet"):

#         with tempfile.NamedTemporaryFile(
#             suffix=".jpg",
#             delete=False
#         ) as temp_file:

#             image.save(temp_file.name)

#             results = model.predict(
#                 source=temp_file.name,
#                 conf=0.25
#             )

#         result = results[0]

#         # Display prediction image
#         plotted_image = result.plot()

#         st.image(
#             plotted_image,
#             caption="Detection Result",
#             use_container_width=True
#         )

#         detected_classes = []

#         for box in result.boxes:
#             class_id = int(box.cls)
#             class_name = str(result.names[class_id])
#             confidence = float(box.conf)

#             detected_classes.append(class_name)

#             st.write(
#                 f"Class: {class_name} | Confidence: {confidence:.2f}"
#             )

#         st.write("Detected Classes:", detected_classes)

#         # model classes:
#         # 0 = No Helmet
#         # 1 = Helmet

#         if "1" in detected_classes:
#             st.success("✅ Helmet Detected")

#         elif "0" in detected_classes:
#             st.error("❌ No Helmet Detected")

#         else:
#             st.warning("⚠ No helmet/person detected")



import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np
import cv2

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Helmet Safety Monitoring System",
    page_icon="🪖",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #00D4FF;
    text-align: center;
}

h3 {
    color: white;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 0px 10px rgba(0,212,255,0.3);
}

.safe {
    color: #00FF7F;
    font-size: 28px;
    font-weight: bold;
}

.danger {
    color: #FF4B4B;
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.title("🪖 AI Helmet Safety Monitoring System")

st.markdown("""
### Real-Time Helmet Compliance Detection using YOLO

This system analyzes uploaded images and determines whether a person is wearing a safety helmet.
""")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("System Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.write("Model Classes:")
st.sidebar.write(model.names)

st.sidebar.write("---")
st.sidebar.write("YOLO Version")
st.sidebar.write("YOLOv11")

# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # -----------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # -----------------------------------
    # DETECT BUTTON
    # -----------------------------------
    if st.button("🚀 Run Detection"):

        with st.spinner("Analyzing Image..."):

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as temp_file:

                image.save(temp_file.name)

                results = model.predict(
                    source=temp_file.name,
                    conf=0.25
                )

        result = results[0]

        plotted = result.plot()

        detected_classes = []
        confidence_scores = []

        for box in result.boxes:

            cls_id = int(box.cls)
            cls_name = str(result.names[cls_id])

            conf = float(box.conf)

            detected_classes.append(cls_name)
            confidence_scores.append(conf)

        max_conf = 0

        if confidence_scores:
            max_conf = max(confidence_scores)

        # -----------------------------------
        # DETECTED IMAGE
        # -----------------------------------
        with col2:
            st.subheader("Detection Result")
            st.image(plotted, use_container_width=True)

        # -----------------------------------
        # STATUS
        # -----------------------------------
        helmet_detected = "1" in detected_classes

        st.write("")

        colA, colB, colC = st.columns(3)

        with colA:

            if helmet_detected:
                st.markdown("""
                <div class="metric-card">
                    <h3>Status</h3>
                    <div class="safe">SAFE ✅</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="metric-card">
                    <h3>Status</h3>
                    <div class="danger">VIOLATION ❌</div>
                </div>
                """, unsafe_allow_html=True)

        with colB:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <h2>{max_conf*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with colC:
            st.markdown("""
            <div class="metric-card">
                <h3>Model</h3>
                <h2>YOLOv11</h2>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------------
        # CONFIDENCE BAR
        # -----------------------------------
        st.write("")
        st.subheader("Detection Confidence")

        st.progress(float(max_conf))

        # -----------------------------------
        # SUMMARY
        # -----------------------------------
        st.subheader("Detection Summary")

        if helmet_detected:
            st.success("Helmet Detected")
        else:
            st.error("Helmet Not Detected")

        st.write("Detected Classes:", detected_classes)

        if confidence_scores:
            st.write(
                "Highest Confidence:",
                f"{max_conf*100:.2f}%"
            )

        # -----------------------------------
        # DOWNLOAD RESULT IMAGE
        # -----------------------------------
        result_image = Image.fromarray(plotted)

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as tmp:

            result_image.save(tmp.name)

            with open(tmp.name, "rb") as file:

                st.download_button(
                    label="📥 Download Result Image",
                    data=file,
                    file_name="helmet_detection_result.png",
                    mime="image/png"
                )

# -----------------------------------
# FOOTER
# -----------------------------------
st.write("---")

st.subheader("📖 About This Project")

st.write("""
This project uses **YOLO-based Object Detection**
to identify helmet compliance in industrial,
construction, and road safety environments.

Applications:

- Construction Site Monitoring
- Factory Safety Systems
- Traffic Safety Enforcement
- PPE Compliance Monitoring

Developed using:
- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
""")