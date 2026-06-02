#!/bin/bash
echo "========================================"
echo "  Magic Gesture 3D Controller"
echo "========================================"
echo ""
echo "Starting server at http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""
cd "$(dirname "$0")"
python server.py
