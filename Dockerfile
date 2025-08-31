# Use official Python slim image with CUDA support for RunPod
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Avoid .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpq-dev \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and wheelhouse
COPY requirements.txt .
COPY wheelhouse/ ./wheelhouse/

# Install packages from wheelhouse
RUN pip install --no-cache-dir --find-links=wheelhouse/ -r requirements.txt

# Copy the Django project
COPY . .

# Collect static files into STATIC_ROOT
RUN python manage.py collectstatic --noinput

# Expose Django port
EXPOSE 8000

# Start the Django app using Gunicorn
CMD ["gunicorn", "football_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers=3", "--timeout=120"]