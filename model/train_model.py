# train_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if not os.path.exists("model/mnist_model.h5"):
    # === Load and preprocess data ===
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # === Define the model ===
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax'),
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # === Train ===
    model.fit(x_train, y_train_cat, epochs=5, validation_split=0.1)

    # === Save model ===
    os.makedirs("model", exist_ok=True)
    model.save("model/mnist_model.h5")

    # === Generate and save confusion matrix ===
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    os.makedirs("images", exist_ok=True)
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")

    print("✅ Model saved to model/mnist_model.h5")
    print("📊 Confusion matrix saved to images/confusion_matrix.png")

else:
    print("✅ Model already exists — skipping training.")
