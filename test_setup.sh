#!/bin/bash

echo "=== AML Investigation Platform - Local Test ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "✅ Python 3 found"

# Check if Node is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found"
    exit 1
fi
echo "✅ Node.js found"

# Test backend setup
echo ""
echo "=== Testing Backend Setup ==="
cd backend

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing Python dependencies..."
pip install -q -r requirements.txt

echo "Generating mock data..."
python generate_mock_data.py

if [ -f "data/elliptic_nodes.parquet" ]; then
    echo "✅ Mock data generated successfully"
else
    echo "❌ Failed to generate mock data"
    exit 1
fi

echo ""
echo "=== Testing Frontend Setup ==="
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
echo "✅ Frontend dependencies installed"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the application:"
echo ""
echo "Terminal 1 (Backend):"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  uvicorn main:app --reload"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo "Then open http://localhost:5173"
