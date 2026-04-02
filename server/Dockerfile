FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Copy source code
COPY src/ src/
COPY server/ server/
COPY configs/ configs/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
