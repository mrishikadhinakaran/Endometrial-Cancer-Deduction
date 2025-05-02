FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your code before downloading the model
COPY . /app

# Pre-download GPT-2
RUN python download_model.py

EXPOSE 7860
CMD ["python", "gradio_app.py"]
