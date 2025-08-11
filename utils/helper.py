import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

MODEL_PATH = "model/cat_dog_model.keras"

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(model, image: Image.Image) -> str:
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    return "Dog" if prediction > 0.5 else "Cat"

def load_cat_dog_model():
    return load_model(MODEL_PATH)