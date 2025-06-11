# Use a base image with Python 3.13
FROM python:3.13-rc-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app runs on
EXPOSE 5002

# Run your Flask app
CMD ["python3", "api.py"]