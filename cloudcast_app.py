import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# Streamlit app for Cloud Motion Prediction
st.set_page_config(page_title="Cloud Motion Predictor", page_icon="☁️", layout="wide")
st.title("☁️ CloudCast: Cloud Motion Predictor")
st.subheader("Upload Satellite Cloud Images (Past Frames)")

# Create a folder to save uploads
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files = st.file_uploader("Upload 3-4 satellite images (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    st.info("Uploaded Images:")
    cols = st.columns(len(uploaded_files))
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        img = Image.open(uploaded_file).resize((256, 256))
        cols[i].image(img, caption=f"Frame {i+1}", use_container_width=True)
        file_paths.append(file_path)

    # Dummy prediction logic (replace with model inference in future)
    if st.button("🔍 Predict Next Cloud Frame"):
        st.subheader("Prediction Result")

        # Load last image as dummy predicted frame
        last_img = cv2.imread(file_paths[-1])
        predicted = cv2.GaussianBlur(last_img, (7, 7), 0)

        actual_img_path = "sample_data/actual_frame.jpeg"  # Replace with real data
        actual_img = cv2.imread(actual_img_path)

        predicted_path = os.path.join(UPLOAD_DIR, "predicted_frame.jpeg")
        cv2.imwrite(predicted_path, predicted)

        col1, col2 = st.columns(2)
        col1.image(predicted_path, caption="Predicted Frame", use_container_width=True)
        col2.image(actual_img_path, caption="Actual Frame", use_container_width=True)

        # Evaluation Metrics
        pred_resized = cv2.resize(predicted, (256, 256))
        actual_resized = cv2.resize(actual_img, (256, 256))

        mae_val = np.mean(np.abs(pred_resized - actual_resized))
        ssim_val = ssim(pred_resized, actual_resized, channel_axis=-1)
        psnr_val = psnr(actual_resized, pred_resized)

        st.markdown("---")
        st.subheader("📊 Evaluation Metrics")
        st.write(f"**SSIM:** {ssim_val:.4f}")
        st.write(f"**PSNR:** {psnr_val:.2f} dB")
        st.write(f"**MAE:** {mae_val:.2f}")

        st.success("Prediction complete. Replace dummy logic with AI model in 6th sem.")

else:
    st.info("Please upload at least 3 satellite image frames to begin prediction.")
