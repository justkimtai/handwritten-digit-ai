import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from digit_recognizer import model

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Resize and normalize
    image = image.resize((28, 28))
    image = np.array(image)
    image = np.invert(image)
    image = image.reshape(1, 784).astype('float32') / 255.0
    
    # Display processed image
    fig, ax = plt.subplots()
    ax.imshow(image.reshape(28, 28), cmap='gray')
    st.pyplot(fig)
    
    # Make prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"Prediction: {predicted_digit}")
    
    st.write("Confidence scores:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{i}: {prob:.2%}")
