FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    nodejs \
    npm \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY apps/reflex_ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/reflex_ui /app

EXPOSE 3000
EXPOSE 8000

CMD ["reflex", "run", "--env", "prod", "--frontend-port", "3000", "--backend-port", "8000", "--backend-host", "0.0.0.0"]