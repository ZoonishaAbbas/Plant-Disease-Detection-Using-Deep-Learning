# CottonGuard AI

**Live app:** [https://plant-disease-detection-using-deep-learning-akytuc2x3vokqvexcv.streamlit.app/](https://plant-disease-detection-using-deep-learning-akytuc2x3vokqvexcv.streamlit.app/)

AI-powered cotton plant disease detection.

Upload a leaf photo and get an instant diagnosis:

- Bacterial Blight
- Curl Virus
- Healthy

Built with Streamlit and TensorFlow Lite.

---

## Features

- Professional dark UI
- Image upload or camera capture
- Prediction with confidence score
- Short treatment guidance per class
- Model hosted on Google Drive (keeps the repo small)

---

## Project structure

```text
cotton-guard/
├── app.py
├── requirements.txt
└── README.md
```

The `.tflite` model is not in this repo. It is downloaded once at runtime.

---

## Dataset

| Split      | Count |
|------------|------:|
| Train      | 903   |
| Test       | 216   |
| Validation | 127   |
| Total      | 1291  |

- Classes: Bacterial Blight, Curl Virus, Healthy
- Image size: 256 x 256 RGB

---

## Local setup

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/cotton-guard.git
cd cotton-guard
```

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Set model URL

1. Upload `cotton_model.tflite` to Google Drive.
2. Share the file as **Anyone with the link** (Viewer).
3. Copy the file id from the share link:

```text
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

4. In `app.py` set:

```python
MODEL_URL = "https://drive.google.com/uc?export=download&id=FILE_ID"
```

### 4. Run

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (public).
2. Open [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**.
4. Select:
   - Repository: your repo
   - Branch: `main`
   - Main file: `app.py`
5. Click **Deploy**.

On the first prediction the app downloads the model from Google Drive. After that it is cached on the server.

---

## Requirements

```text
streamlit>=1.32.0
tensorflow-cpu>=2.15.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## Model

| Item        | Detail                          |
|-------------|---------------------------------|
| Format      | TensorFlow Lite (quantized)     |
| Input       | UINT8, 256 x 256                |
| Size        | about 100 MB                    |
| Hosting     | Google Drive (not in GitHub)    |

If Google Drive blocks the download, upload the same file to Hugging Face and update `MODEL_URL`.

---

## How it works

1. User uploads or captures a leaf image.
2. Image is resized to 256 x 256.
3. TFLite model runs inference.
4. App shows class, confidence, and a short tip.

---

## Disclaimer

For educational and screening use only. Confirm important decisions with agronomy experts.

---

## License

Free to use and modify for learning and research.
