# 1. Start from official Python image
FROM python:3.12-slim

# 2. Environment settings for production-like behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set working directory inside container
WORKDIR /app

# 4. Copy requirements and install Python dependencies
COPY rag/requirements.txt .
RUN apt-get update && \
    apt-get install -y git && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# 5. Copy app source code into container
COPY rag/ ./

# 6. Ensure uploads folder exists and is writable
RUN mkdir -p uploads && chmod -R 777 uploads

# 7. Expose Flask app port
EXPOSE 5000

# 8. Start Flask
# CMD ["python", "osama_rag.py"]
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "osama_rag:app"]