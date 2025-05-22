import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model("mnist_cnn_model.h5")

st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload an image of a digit (0-9), and the model will predict what it is.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    # Convert to grayscale and resize
    image = image.convert("L")
    image = ImageOps.invert(image)  # Invert to match MNIST style
    image = image.resize((28, 28))
    
    # Convert to array
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)

        st.success(f"🧾 Predicted Digit: **{predicted_label}**")
