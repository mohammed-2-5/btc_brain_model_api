# ==== Base image ====
FROM python:3.11-slim

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*

# ==== Python deps ====
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==== Copy code & model ====
COPY app/ .

# ==== Expose & run ====
ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
