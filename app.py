import os
import urllib.request
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CottonGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- MODEL URL  ----------------
# 1. Upload cotton_model.tflite to Google Drive

MODEL_URL = "https://drive.google.com/uc?export=download&id=10tHaRXeBh-620dwpBYkAFRmijrthBW4U"
MODEL_PATH = "cotton_model.tflite"


# ---------------- STYLES ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg-deep: #07110c;
  --bg-mid: #0c1a14;
  --surface: rgba(16, 36, 28, 0.88);
  --surface-2: rgba(22, 48, 38, 0.82);
  --border: rgba(110, 180, 140, 0.22);
  --text: #e8f5ee;
  --text-muted: #9fb8aa;
  --accent: #2dd4a0;
  --radius: 16px;
  --shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(900px 500px at 10% -10%, rgba(45, 212, 160, 0.12), transparent 60%),
    radial-gradient(800px 480px at 90% 0%, rgba(52, 211, 153, 0.08), transparent 55%),
    linear-gradient(180deg, var(--bg-mid), var(--bg-deep));
}

header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 1.2rem; max-width: 1180px; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1611, #07110c);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label {
  background: rgba(16, 36, 28, 0.6);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 14px;
  margin-bottom: 8px;
  width: 100%;
}
[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }
[data-testid="stSidebar"] .stRadio label div,
[data-testid="stSidebar"] .stRadio label span {
  color: var(--text) !important;
  font-weight: 600 !important;
}
.brand {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.55rem;
  font-weight: 700;
  color: #ecfdf5;
  margin-bottom: 4px;
}
.brand-sub { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 20px; }
.hero {
  background: linear-gradient(135deg, rgba(16, 42, 32, 0.95), rgba(12, 32, 24, 0.9));
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 36px 40px;
  box-shadow: var(--shadow);
  margin-bottom: 28px;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.35rem;
  font-weight: 700;
  color: #f0fdf4;
  line-height: 1.2;
  margin: 0 0 12px 0;
}
.hero-lead {
  color: var(--text-muted);
  font-size: 1.08rem;
  line-height: 1.65;
  max-width: 640px;
  margin: 0;
}
.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.35rem;
  font-weight: 700;
  color: #ecfdf5;
  margin: 8px 0 16px 0;
}
.card {
  background: linear-gradient(160deg, var(--surface), var(--surface-2));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
  height: 100%;
}
.card-icon { font-size: 1.6rem; margin-bottom: 10px; }
.card-title { font-weight: 700; font-size: 1.05rem; color: #f0fdf4; margin-bottom: 8px; }
.card-body { color: var(--text-muted); font-size: 0.95rem; line-height: 1.55; margin: 0; }
.stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 18px 0 8px 0; }
.stat {
  background: rgba(16, 36, 28, 0.7);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
  min-width: 120px;
  flex: 1;
}
.stat-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--accent);
}
.stat-label { color: var(--text-muted); font-size: 0.82rem; margin-top: 4px; }
.badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
}
.badge-danger { background: rgba(239, 68, 68, 0.18); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.35); }
.badge-warn { background: rgba(245, 158, 11, 0.18); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }
.badge-ok { background: rgba(45, 212, 160, 0.18); color: #6ee7b7; border: 1px solid rgba(45, 212, 160, 0.35); }
.result-box {
  background: linear-gradient(160deg, rgba(16, 42, 32, 0.95), rgba(12, 32, 24, 0.9));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px 28px;
  margin-top: 20px;
  text-align: center;
}
.result-label { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 8px; }
.result-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.75rem;
  font-weight: 700;
  color: #f0fdf4;
  margin-bottom: 6px;
}
.result-hint {
  color: var(--text-muted);
  font-size: 0.95rem;
  line-height: 1.5;
  max-width: 480px;
  margin: 10px auto 0 auto;
}
.sample-caption { text-align: center; color: var(--text-muted); font-size: 0.9rem; margin-top: 8px; font-weight: 600; }
.footer-note {
  color: var(--text-muted);
  font-size: 0.85rem;
  text-align: center;
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
}
.stButton > button {
  background: linear-gradient(90deg, #2dd4a0, #34d399) !important;
  color: #052e1c !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  padding: 0.6rem 1.4rem !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: rgba(16, 36, 28, 0.7) !important;
  border: 1.5px dashed rgba(45, 212, 160, 0.45) !important;
  border-radius: 14px !important;
}
.stRadio label p, .stSelectbox label p { color: #ecfdf5 !important; font-weight: 600 !important; }
div[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--border); }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- MODEL (download + TFLite) ----------------
def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return

    if "YOUR_FILE_ID_HERE" in MODEL_URL:
        raise FileNotFoundError(
            "Set MODEL_URL with your Google Drive file id before deploying."
        )

    with st.spinner("Downloading model (first run only, ~100 MB)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        with open(MODEL_PATH, "rb") as f:
            head = f.read(100)

        # Handle Google Drive virus-scan confirm page for large files
        if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html"):
            import re

            with open(MODEL_PATH, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
            file_id_match = re.search(r"id=([^&]+)", MODEL_URL)
            if match and file_id_match:
                token = match.group(1)
                file_id = file_id_match.group(1)
                confirm_url = (
                    f"https://drive.google.com/uc?export=download"
                    f"&confirm={token}&id={file_id}"
                )
                urllib.request.urlretrieve(confirm_url, MODEL_PATH)

        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
            raise RuntimeError(
                "Model download failed. Make the Drive file public and check MODEL_URL."
            )


@st.cache_resource
def load_tflite_model():
    download_model()
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def model_prediction(test_image):
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if hasattr(test_image, "read"):
        image = Image.open(test_image).convert("RGB")
        try:
            test_image.seek(0)
        except Exception:
            pass
    else:
        image = Image.open(test_image).convert("RGB")

    image = image.resize((256, 256))
    arr = np.array(image)

    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.uint8:
        arr = np.expand_dims(arr.astype(np.uint8), axis=0)
    elif input_dtype == np.int8:
        arr = np.expand_dims(arr.astype(np.int8), axis=0)
    else:
        arr = np.expand_dims(arr.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    preds = np.array(preds, dtype=np.float32)

    if preds.max() > 1.5:
        preds = preds / (preds.sum() + 1e-8)

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    if confidence > 1.0:
        confidence = min(confidence / 100.0, 1.0) if confidence > 10 else min(confidence, 1.0)

    return idx, confidence


CLASS_INFO = {
    "bacterial_blight": {
        "label": "Bacterial Blight",
        "badge": "badge-danger",
        "hint": "Water-soaked lesions and angular leaf spots are common. Remove infected leaves, avoid overhead irrigation, and consider copper-based treatments.",
    },
    "curl_virus": {
        "label": "Curl Virus",
        "badge": "badge-warn",
        "hint": "Leaves curl and growth may be stunted. Control whitefly vectors, remove infected plants early, and use resistant varieties when possible.",
    },
    "healthy": {
        "label": "Healthy",
        "badge": "badge-ok",
        "hint": "No disease symptoms detected. Keep monitoring regularly and maintain good field hygiene for continued plant health.",
    },
}


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="brand">CottonGuard AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Cotton plant disease detection</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Home", "Disease Recognition", "About"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Powered by deep learning · 256×256 RGB")


# ---------------- HOME ----------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">Detect cotton plant diseases in seconds</div>
          <p class="hero-lead">
            Upload a leaf photo and get an instant AI diagnosis — Bacterial Blight,
            Curl Virus, or Healthy — so you can act early and protect your crop.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        st.image("image.jpg", use_container_width=True)
    except Exception:
        pass

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><div class="card-icon">1</div><div class="card-title">Upload or capture</div><p class="card-body">Take a clear photo of the leaf or upload an existing image from your device.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-icon">2</div><div class="card-title">AI analysis</div><p class="card-body">A trained deep learning model examines visual symptoms at 256×256 resolution.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div class="card-icon">3</div><div class="card-title">Diagnosis & tips</div><p class="card-body">See the predicted class with a short practical recommendation for next steps.</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why use CottonGuard</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><div class="card-title">High accuracy</div><p class="card-body">Trained on labeled cotton leaf images across three classes for reliable field screening.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Simple workflow</div><p class="card-body">No complex setup — open the app, add a photo, and get a result in moments.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div class="card-title">Early action</div><p class="card-body">Spot issues sooner so you can treat, isolate, or adjust practices before losses grow.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="footer-note">Go to <strong>Disease Recognition</strong> in the sidebar to start.</div>', unsafe_allow_html=True)


# ---------------- DISEASE RECOGNITION ----------------
elif page == "Disease Recognition":
    st.markdown('<div class="section-title">Disease Recognition</div>', unsafe_allow_html=True)
    st.markdown("Upload a leaf image or capture one with your camera. The model classifies **Bacterial Blight**, **Curl Virus**, or **Healthy**.")

    option = st.radio("Image source", ["Upload Image", "Capture from Camera"], horizontal=True)
    test_image = None
    if option == "Upload Image":
        test_image = st.file_uploader("Upload a cotton leaf image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    else:
        test_image = st.camera_input("Capture leaf photo", label_visibility="collapsed")

    if test_image is not None:
        col_img, col_action = st.columns([1.2, 1])
        with col_img:
            st.image(test_image, caption="Selected image", use_container_width=True)
        with col_action:
            st.markdown("#### Ready to analyze")
            st.caption("Click predict to run the model on this image.")
            if st.button("Predict disease", use_container_width=True):
                with st.spinner("Analyzing leaf image..."):
                    try:
                        idx, confidence = model_prediction(test_image)
                        class_keys = ["bacterial_blight", "curl_virus", "healthy"]
                        key = class_keys[idx]
                        info = CLASS_INFO[key]
                        st.markdown(
                            f"""
                            <div class="result-box">
                              <div class="result-label">Model prediction</div>
                              <div class="result-name">{info['label']}</div>
                              <span class="badge {info['badge']}">{key.replace('_', ' ')}</span>
                              <p class="result-hint">{info['hint']}</p>
                              <p class="result-hint" style="margin-top:12px;opacity:0.85">
                                Confidence: <strong>{confidence * 100:.1f}%</strong>
                              </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    except FileNotFoundError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload or capture a leaf image to begin.")


# ---------------- ABOUT ----------------
elif page == "About":
    st.markdown('<div class="section-title">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card" style="margin-bottom: 22px;">
          <div class="card-title">Dataset</div>
          <p class="card-body">
            1,291 images of cotton plants across three classes: Bacterial Blight, Curl Virus, and Healthy.
            Each image is 256×256 pixels with RGB channels and is used to train a deep learning classifier
            that screens leaf health from visual symptoms.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="stat-row">
          <div class="stat"><div class="stat-value">903</div><div class="stat-label">Train images</div></div>
          <div class="stat"><div class="stat-value">216</div><div class="stat-label">Test images</div></div>
          <div class="stat"><div class="stat-value">127</div><div class="stat-label">Validation</div></div>
          <div class="stat"><div class="stat-value">3</div><div class="stat-label">Classes</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classes</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><span class="badge badge-danger">Disease</span><div class="card-title" style="margin-top:10px">Bacterial Blight</div><p class="card-body">Lesions on leaves and stems; water-soaked angular spots are typical.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><span class="badge badge-warn">Disease</span><div class="card-title" style="margin-top:10px">Curl Virus</div><p class="card-body">Curled, deformed leaves and stunted growth from viral infection.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><span class="badge badge-ok">Healthy</span><div class="card-title" style="margin-top:10px">Healthy</div><p class="card-body">Normal growth and leaf development with no disease symptoms.</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sample images</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            st.image("bact149.png", use_container_width=True)
            st.markdown('<div class="sample-caption">Bacterial Blight</div>', unsafe_allow_html=True)
        except Exception:
            st.caption("bact149.png not found")
    with col2:
        try:
            st.image("curl83.jpg", use_container_width=True)
            st.markdown('<div class="sample-caption">Curl Virus</div>', unsafe_allow_html=True)
        except Exception:
            st.caption("curl83.jpg not found")
    with col3:
        try:
            st.image("h402.jpg", use_container_width=True)
            st.markdown('<div class="sample-caption">Healthy leaf</div>', unsafe_allow_html=True)
        except Exception:
            st.caption("h402.jpg not found")

    st.markdown('<div class="footer-note">CottonGuard AI · for educational and screening use · always confirm critical decisions with agronomy experts</div>', unsafe_allow_html=True)
