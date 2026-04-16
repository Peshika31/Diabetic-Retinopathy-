import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import gdown
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# 🎨 UI CONFIG + STYLE
# =======================
st.set_page_config(page_title="DR Detection AI", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00FFAA;
}
.subtitle {
    text-align: center;
    color: #AAAAAA;
    margin-bottom: 30px;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 DR Detection AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Lesion-Aware | Confidence-Based | Clinical Decision Support</div>', unsafe_allow_html=True)

# =======================
# CBAM
# =======================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.fc(x)
        attn = self.sigmoid(attn)
        return x * attn

# =======================
# DUAL MODEL
# =======================
class DualBranchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone1 = models.efficientnet_b0(weights=None)
        self.backbone2 = models.efficientnet_b0(weights=None)

        self.backbone1.classifier = nn.Identity()
        self.backbone2.classifier = nn.Identity()

        self.cbam1 = CBAM(1280)
        self.cbam2 = CBAM(1280)

        self.fc = nn.Sequential(
            nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x1, x2):
        f1 = self.backbone1(x1)
        f2 = self.backbone2(x2)

        f1 = self.cbam1(f1)
        f2 = self.cbam2(f2)

        fused = torch.cat((f1, f2), dim=1)
        return self.fc(fused)

# =======================
# LOAD MODEL
# =======================
@st.cache_resource
def load_model():
    model_path = "Best_model.pth"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1yZIfaPk2hlUVDDVyUg1lSmE_Lbe4ujdw"
        gdown.download(url, model_path, quiet=False)

    model = DualBranchModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# =======================
# PREPROCESS FUNCTIONS
# =======================

# 🔹 Baseline (your CLAHE)
def baseline_preprocess(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def preprocess_baseline(image):
    image = np.array(image)
    image = baseline_preprocess(image)
    image = cv2.resize(image, (224,224))
    image = image / 255.0

    image = np.transpose(image, (2,0,1))
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32).to(device)


# 🔹 Lesion preprocess (IMPORTANT)
def preprocess_lesion(image):
    image = np.array(image)
    image = cv2.resize(image, (224,224))

    green = image[:,:,1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)

    enhanced = np.stack([enhanced]*3, axis=-1)

    enhanced = enhanced / 255.0
    enhanced = np.transpose(enhanced, (2,0,1))
    enhanced = np.expand_dims(enhanced, axis=0)

    return torch.tensor(enhanced, dtype=torch.float32).to(device)


# =======================
# LAYOUT
# =======================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload Fundus Image")
    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")

with col2:
    st.markdown("### 📊 Prediction Result")

    if uploaded_file:

        img1 = preprocess_baseline(image)
        img2 = preprocess_lesion(image)

        with st.spinner("Analyzing Image..."):
            with torch.no_grad():
                output = model(img1, img2)
                prob = torch.sigmoid(output).item()

        prediction = "DR Detected" if prob > 0.5 else "No DR"
        confidence = prob if prob > 0.5 else 1 - prob
        uncertainty = 1 - abs(prob - 0.5)*2

        color = "#FF4B4B" if prediction == "DR Detected" else "#00FFAA"

        st.markdown(f"""
        <div class="card">
            <h2 style="color:{color};">{prediction}</h2>
            <p><b>Confidence:</b> {confidence:.2f}</p>
            <p><b>Uncertainty:</b> {uncertainty:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence * 100))

        st.caption("🧠 Uncertainty indicates how unsure the model is. Lower = more reliable.")

    else:
        st.info("Upload an image to see prediction")

# =======================
# FOOTER
# =======================
st.markdown("---")
st.markdown("Developed for Diabetic Retinopathy Screening 🚀")