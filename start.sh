#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d "venv" ]; then
  echo "venv not found — run: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
  exit 1
fi
source venv/bin/activate
( sleep 2 && (xdg-open http://localhost:5000 >/dev/null 2>&1 || true) ) &
echo "Starting World Cup Predictor — http://localhost:5000"
echo "Press Ctrl+C to stop. Nothing will keep running after that."
python3 server.py
