FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# --no-cache-dir reduces image size by skipping pip's download cache
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY main.py .
COPY best_model/ ./best_model/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
