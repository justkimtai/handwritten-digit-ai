# digit_recognizer.py

import tensorflow as tf
import os

# Load the trained model from the 'models' directory
model_path = os.path.join("models", "mnist_model.h5")
model = tf.keras.models.load_model(model_path)
