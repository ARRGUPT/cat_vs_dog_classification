# 🐱 Cat vs 🐶 Dog Classification (CNN + Streamlit)

A simple image classification project that predicts whether an uploaded image is of a **cat** or a **dog**, built with **Keras**, **TensorFlow**, and **Streamlit**.

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd cat_vs_dog_classification
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App
```bash
streamlit run app.py
```
Open the URL shown in your terminal (usually `http://localhost:8501`) in your browser.

---

## 📌 Features
- Upload an image (`.jpg`, `.jpeg`, `.png`)
- Model predicts **Cat** or **Dog**

---

## 📊 Model Info
- **Architecture**: CNN with Conv2D, BatchNorm, MaxPooling, Dense layers
- **Training Data**: Kaggle Dogs vs Cats dataset
- **Frameworks**: TensorFlow + Keras

---

## 📈 Results
The training history and final evaluation metrics are provided to assess the model's performance in classifying dog and cat images.

**Test Accuracy**: `88.04%`

---

## 📝 License
This project is for learning purposes and can be modified freely.