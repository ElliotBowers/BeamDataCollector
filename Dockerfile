FROM python:3.11-slim

# Install system dependencies for building the C++ extension
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3.11-dev \
    git \
    libssl-dev \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy your pydapter project
COPY pydapter-wraptor /app/pydapter-wraptor

# Install Python dependencies for building (after system pybind11 is installed)
RUN pip install --upgrade pip setuptools wheel

# This line will now work since pybind11 headers are available
RUN pip install /app/pydapter-wraptor

# Copy rest of your app
COPY BeamDataCollector.py .
COPY main.py .
COPY config.yml .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python", "main.py"]
