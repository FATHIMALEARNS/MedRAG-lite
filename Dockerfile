# Use official Python lightweight image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install minimal system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and Torchvision specifically using the CPU wheel to keep the container size small
# (The default PyPi version includes CUDA, which adds ~2GB to the image size)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy only the requirements file first (this caches the pip install step in Docker)
COPY requirements.txt .

# Install the rest of the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your application code into the container
COPY . .

# Set environment variables so Python doesn't buffer standard output
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port 5000 (the port Flask runs on)
EXPOSE 5000

# Start the Flask app bound to all interfaces (0.0.0.0) so it's accessible outside the container
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
