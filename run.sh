#!/bin/bash

# HRHUB Quick Start Script

echo "🚀 Starting HRHUB..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

echo ""
echo "🎉 Launching Streamlit app..."
echo "📍 Open your browser to: http://localhost:8501"
echo ""

streamlit run app.py
