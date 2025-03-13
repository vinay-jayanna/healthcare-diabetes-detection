# Use official Python image
FROM python:3.11.2

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt /app/

# Install required dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy model and runtime files
COPY diabetic_retinopathy_model.h5 /app/model/
COPY custom_runtime.py /app/
COPY settings.json /app/model-settings.json

# Set MLServer environment variables
ENV MLSERVER_MODELS_DIR=/app
ENV MLSERVER_MODEL_NAME=diabetic_retinopathy_model

# Expose MLServer's default port
EXPOSE 8080

# Run MLServer
CMD ["mlserver", "start", "/app"]
