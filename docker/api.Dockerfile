FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir uv
RUN uv pip install --system .

EXPOSE 8000

CMD ["uvicorn", "baikpacking.api.main:app", "--host", "0.0.0.0", "--port", "8000"]