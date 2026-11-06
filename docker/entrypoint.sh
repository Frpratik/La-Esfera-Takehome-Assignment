#!/bin/sh
# Start the API, building the index first if it is missing.
set -e

INDEX_DIR="${INDEX_DIR:-data/index}"

if [ ! -f "$INDEX_DIR/metadata.json" ]; then
    echo "index not found in $INDEX_DIR, building it..."
    python -m ingest.build_index
fi

# exec so uvicorn becomes PID 1 and receives SIGTERM from `docker stop`.
exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
