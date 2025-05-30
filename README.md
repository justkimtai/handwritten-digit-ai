# 🔢 MNIST Streamlit Classifier

An interactive Streamlit app that uses a trained TensorFlow model to recognize handwritten digits from the MNIST dataset.

## 🚀 Live Demo

👉 [Try the live app here](https://breastcancerclassifierdemo.streamlit.app/)

## 🎯 Features

- Predicts handwritten digits from uploaded 28×28 grayscale images
- Visualizes prediction probabilities as a bar chart
- Built-in preprocessing pipeline (inversion, normalization, flattening)
- Uses a trained multi-layer neural network (MLP with dropout)
- Streamlit interface for clean, responsive interactivity
- Ready for local development or public cloud deployment


## 🛠 Tech Stack

- Python 3
- TensorFlow
- Streamlit
- NumPy
- PIL (Python Imaging Library)

## 📂 Dataset

- MNIST Dataset: 70,000 images of handwritten digits
- Automatically downloaded via tensorflow.keras.datasets.mnist

## 📸 Screenshots

| Image Upload         | Model in Operation              | Prediction Result            |
|----------------------|---------------------------------|------------------------------|
| ![UI](images/3.jpeg) | ![MO](images/model_running.png) | ![PR](images/prediction.png) |

## 🧪 Getting Started Locally

```bash
# Clone the repo
git clone https://github.com/justkimtai/handwritten-digit-ai.git
cd handwritten-digit-ai

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 🤝 Credits

This project was inspired by [DigitalOcean tutorials](https://www.digitalocean.com/community/tutorials) and built as part of my machine learning learning journey.

## 📩 Contact

Feel free to connect with me on [X (Twitter)](https://x.com/justkimtai) or [email](mailto:justkimtai@gmail.com) me for collaboration, freelance work, or opportunities!
