FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Launch Uvicorn server on port 7860
CMD ["python", "main.py"]
