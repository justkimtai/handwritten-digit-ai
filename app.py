# app.py

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from digit_recognizer import model

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")
st.title("🧠 Handwritten Digit Recognition")
st.markdown("Upload an image of a handwritten digit (0–9)")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the original uploaded image
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((28, 28))  # MNIST size
    image = np.array(image)
    image = np.invert(image)  # Invert image: white background, black digit
    image = image.reshape(1, 784).astype('float32') / 255.0

    # Visualize the processed image
    st.markdown("### 🧪 Processed Input")
    fig, ax = plt.subplots()
    ax.imshow(image.reshape(28, 28), cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    # Predict
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    st.success(f"✅ Prediction: **{predicted_digit}**")

    st.markdown("### 🔢 Confidence Scores")
    for i, prob in enumerate(prediction[0]):
        st.write(f"**{i}**: {prob:.2%}")
