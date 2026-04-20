#!/usr/bin/env bash
set -euo pipefail

docker compose down -v
docker compose up -d postgres

echo "Waiting for Postgres to become healthy..."
until docker exec baikpacking-postgres pg_isready -U baikpacking -d baikpacking >/dev/null 2>&1; do
  sleep 2
done

echo "Verifying pgvector extension..."
docker exec -i baikpacking-postgres psql -U baikpacking -d baikpacking -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"

echo "Restoring backup..."
docker exec -i baikpacking-postgres pg_restore -U baikpacking -d baikpacking /backups/baikpacking.dump

echo "Done."