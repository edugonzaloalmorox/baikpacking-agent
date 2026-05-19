#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${OLLAMA_MODEL:-mxbai-embed-large:335m}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/opt/ollama-models}"
export OLLAMA_HOST="127.0.0.1:11434"

echo "OLLAMA_MODELS=${OLLAMA_MODELS}"
echo "Starting Ollama server..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama HTTP API..."
for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama is ready."
    break
  fi

  if ! kill -0 "$OLLAMA_PID" >/dev/null 2>&1; then
    echo "Ollama exited early. Logs:"
    cat /tmp/ollama.log || true
    exit 1
  fi

  sleep 1
done

echo "Pulling model without streaming: ${MODEL_NAME}"
curl -fsS --max-time 1800 http://127.0.0.1:11434/api/pull \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${MODEL_NAME}\",\"stream\":false}"

echo
echo "Checking models after pull:"
curl -fsS http://127.0.0.1:11434/api/tags

echo
echo "Model files:"
find "$OLLAMA_MODELS" -maxdepth 5 -type f | head -50

echo
echo "Stopping Ollama..."
kill "$OLLAMA_PID" || true
wait "$OLLAMA_PID" || true
