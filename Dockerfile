# Use a base image with Python 3.13
FROM python:3.13-rc-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.ollama/bin:$PATH"

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    gnupg \
    software-properties-common \
    && apt-get clean

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pre-pull a model (optional but recommended to avoid latency at runtime)
RUN ollama run llama3 --help || true

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose ports: Flask (5002) and Ollama (11434)
EXPOSE 5002 11434

# Start Ollama and Flask app in parallel
CMD ollama serve & python3 api.py