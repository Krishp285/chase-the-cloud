# ☁️ Chase the Cloud – Cloud Motion Prediction App

**Chase the Cloud** is a cloud motion forecasting application using satellite image sequences. This project is part of the **Design Engineering subject (5th–6th Sem)** under GTU and will be extended with AI/ML (Diffusion Models) in the next semester.

---

## 🎯 Project Objective

To develop a prototype for **short-term (0–3 hours)** cloud motion prediction using:
- Indian satellite data (INSAT-3DR/3DS)
- Spatio-temporal image sequences (IR, VIS, WV)
- **Diffusion models** in the next phase

---

## 📸 App Features

- Upload **3–4 satellite cloud images**
- Displays predicted cloud motion (dummy frame for now)
- Compare against actual image
- Shows **SSIM, PSNR, MAE** metrics
- Web UI built with **Streamlit**
- Will integrate trained **AI model** in semester 6

---

## 📁 Folder Structure

chase-the-cloud/
├── cloudcast_app.py # Streamlit app code
├── requirements.txt # Streamlit dependencies
├── sample_data/
│ ├── predicted_frame.jpeg
│ └── actual_frame.jpeg


---

## 🚀 Run Locally

```bash
git clone https://github.com/Krishp285/chase-the-cloud.git
cd chase-the-cloud
pip install -r requirements.txt
streamlit run cloudcast_app.py
