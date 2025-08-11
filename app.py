import streamlit as st
from utils.helper import load_cat_dog_model, predict_image
from PIL import Image

st.title("🐱 Cat vs 🐶 Dog Classifier")

model = load_cat_dog_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png","avif"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        label = predict_image(model, image)
        st.success(f"Its a: **{label}**")
