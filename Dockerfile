# Use a slim official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install OS dependencies for scientific libraries and Streamlit
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files into container
COPY . .

# Train the model during build
RUN python model/train_model.py

# Expose the Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
