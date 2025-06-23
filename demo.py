import os
import base64
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st
from PIL import Image
from io import BytesIO

# Set your API Key
GOOGLE_API_KEY = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- AGENTS ---
info_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

summary_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    markdown=True
)

class MonitorAgent:
    def __init__(self):
        self.logs = []

    def log(self, source, content):
        self.logs.append((source, content))

monitor_agent = MonitorAgent()

# -- Analyze image function --
def analyze_medical_image(image_path):
    image = PILImage.open(image_path)
    aspect_ratio = image.width / image.height
    new_height = int(500 / aspect_ratio)
    resized_image = image.resize((500, new_height))

    temp_path = "temp_resized_image.png"
    resized_image.save(temp_path)

    agno_image = AgnoImage(filepath=temp_path)

    try:
        # Websearch + Image analysis
        info_prompt = """
        You are a highly skilled medical information agent. Based on the uploaded radiology image,
        provide potential diagnoses, related symptoms, and any clinically relevant findings or medical literature insights.
        """
        info_response = info_agent.run(info_prompt, images=[agno_image])
        info_text = info_response.content
        monitor_agent.log("Websearch & Info Agent", info_text)

        # Generate short summary
        summary_prompt = f"""
        Write a clear and concise medical summary in 30 to 40 words from the following details:

        {info_text}
        """
        summary_response = summary_agent.run(summary_prompt)
        summary_text = summary_response.content
        monitor_agent.log("Summary Agent", summary_text)

        return info_text, summary_text

    except Exception as e:
        return f"⚠️ Analysis error: {e}", ""
    finally:
        agno_image.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Streamlit UI ---
st.set_page_config(page_title="Medical Image Analysis Agent", layout="centered")

# Add Google Font for Poppins
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        body { font-family: 'Poppins', sans-serif; }
        [data-testid="stSidebar"] {
            width: 320px !important;
            background: linear-gradient(to bottom right, #003D62, #3497BA);
            color: white;
            padding: 20px;
        }
        h1, h2, h3, h4, h5, h6, p {
            font-family: 'Poppins', sans-serif;
        }
        .custom-button {
            background-color: #2f729b;
            color: white;
            padding: 0.6em 1.2em;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .custom-button:hover {
            background-color: #255d7a;
        }
        .stButton>button {
            background-color: #2f729b !important;
            color: white !important;
            border-radius: 8px;
            font-size: 16px;
            height: 45px;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# -- 👨‍⚕️ Title & Introduction
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 30px; color: #2f729b; margin-bottom: 20px;'>
            Medical Image Analysis Agent
        </h1>
        <p>
            Welcome to the <strong>Medical Image Analysis Agent</strong>! <br>
            Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and our AI-powered system will analyze it,<br>
            providing detailed findings, diagnosis, and research insights.<br>
            Let's get started!
        </p>
    </div>
""", unsafe_allow_html=True)

# -- 🖼 Logo with Rounded Corners in Sidebar
def get_base64_logo(img_path):
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.sidebar.error(f"Logo error: {e}")
        return None

logo_base64 = get_base64_logo("static/Hoonartek-V25-White-Color.png")  # Update path as needed

if logo_base64:
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 220px; border-radius: 0px; margin-bottom: 20px;" />
        </div>
        """,
        unsafe_allow_html=True
    )

# -- 🧠 Use Case Description
st.sidebar.title("Use Case Details")
st.sidebar.markdown(
    """
    This module uses Gemini 2.0 Flash and Agno Agents to analyze radiology images, delivering expert AI insights for diagnostics and screening.
    """
)
st.sidebar.header("Model Use:")
st.sidebar.markdown("Gemini 2.0 Flash")

# -- 📤 Upload Image
uploaded_file = st.sidebar.file_uploader("Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.sidebar.button("Analyze Image"):
        with st.spinner("Analyzing the image... Please wait."):
            image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            info_output, summary_output = analyze_medical_image(image_path)

            st.subheader("Report Analysis")
            st.markdown(info_output, unsafe_allow_html=True)

            st.subheader("Summary")
            st.markdown(summary_output, unsafe_allow_html=True)

            os.remove(image_path)
