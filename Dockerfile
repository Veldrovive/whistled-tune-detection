FROM python:3.11-slim

# Install system dependencies
# portaudio19-dev is needed for PyAudio
# libsndfile1 is needed for librosa
# gcc is needed for building some python packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# We install these directly to avoid needing a requirements.txt for this simple setup,
# but using requirements.txt is also good practice.
RUN pip install --no-cache-dir \
    python-kasa \
    pyaudio \
    numpy \
    scipy \
    librosa

# Copy project files
COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "kasa_light_controller.py"]
