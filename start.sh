#!/bin/bash
echo "========================================"
echo "  Magic Gesture 3D Controller - Dual Hand"
echo "========================================"
echo ""
echo "Server will auto-find available port (see console output)"
echo "Press Ctrl+C to stop"
echo ""
cd "$(dirname "$0")"
python server.py
