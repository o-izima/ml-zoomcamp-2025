FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy files
COPY pyproject.toml .
COPY app.py .
COPY pipeline_v1.bin .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn==1.6.1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
