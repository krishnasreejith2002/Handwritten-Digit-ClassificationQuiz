import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set up Streamlit app
st.set_page_config(page_title="Digit Classifier", layout="centered")

st.title("🧠 Handwritten Digit Classifier")
st.markdown("Upload a 28x28 image of a digit (0-9) to classify it using a trained model.")

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("digit_model.h5")
    return model

model = load_model()

# Image uploader
uploaded_file = st.file_uploader("Upload your digit image", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    img = img.convert('L')  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)
    
    # Preprocess and predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    pred_class = np.argmax(prediction)

    st.subheader(f"🧾 Prediction: {pred_class}")
    st.bar_chart(prediction[0])
