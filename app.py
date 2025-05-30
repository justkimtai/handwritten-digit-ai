# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model/mnist_model.h5")

# Preprocess uploaded image
def preprocess_image(image):
    image = image.convert("L").resize((28, 28))  # Grayscale + Resize
    img_array = np.asarray(image)
    img_array = 255 - img_array  # Invert
    img_array = img_array / 255.0  # Normalize
    return img_array.reshape(1, 28, 28)

st.title("🔢 MNIST Digit Classifier")
st.write("Upload a 28×28 image of a handwritten digit (white on black) to predict the digit.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {predicted_class}")

    # Display probability bar chart
    st.write("Prediction Probabilities:")
    st.bar_chart(prediction[0])

    # Save prediction chart
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/predicted_class_distribution.png")
    st.success("Prediction chart saved to images/predicted_class_distribution.png")
