
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/train models

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]




# ============================================
# FILE: 
# ============================================


# ============================================
# FILE: kubernetes/deployment.yaml
# ============================================



# ============================================
# FILE: 
# ============================================



# ============================================
# FILE: deploy.sh
# ============================================


# ============================================
# FILE: test_api.sh
# ============================================


# ============================================
# FILE: setup_gcp.sh (Google Cloud Platform)
# ============================================


# ============================================
# FILE: setup_aws.sh (AWS)
# ============================================



# ============================================
# FILE: CI_CD/.github/workflows/deploy.yml
# ============================================
