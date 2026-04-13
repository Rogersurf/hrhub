# FORCE REBUILD
FROM python:3.10-slim

# Avoid Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# HF PRE-BUILD SAFE DEPENDENCIES
# ---------------------------------------------------------
# Hugging Face runs a pre-build pip install on requirements.txt
# using Python 3.13 BEFORE this Dockerfile.
# Therefore, requirements.txt MUST stay minimal and compatible.
# ---------------------------------------------------------

# Copy minimal requirements first (for HF pre-build + Docker cache)
COPY requirements.txt .

# Upgrade build tooling and install minimal dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# APPLICATION DEPENDENCIES (INSTALLED ONLY INSIDE DOCKER)
# ---------------------------------------------------------
# These dependencies are NOT visible to HF pre-build.
# They are installed here with Python 3.10 as intended.
# ---------------------------------------------------------

# Copy application-specific requirements
COPY requirements_app.txt .

# Install full scientific / application stack
RUN pip install --no-cache-dir -r requirements_app.txt

# ---------------------------------------------------------
# APPLICATION CODE
# ---------------------------------------------------------

# Copy application code
COPY app.py .
COPY pages ./pages
COPY utils ./utils

# ---------------------------------------------------------
# STREAMLIT CONFIGURATION
# ---------------------------------------------------------

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose port for HF
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
