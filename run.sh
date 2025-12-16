#!/bin/bash
echo "🚀 Starting HRHUB..."
echo ""

# Try Python 3.10, then 3.11, then default
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
    echo "✅ Python 3.10 found (Hugging Face compatible)"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "⚠️  Python 3.11 found (almost compatible)"
else
    PYTHON_CMD="python3"
    echo "⚠️  Using default Python: $($PYTHON_CMD --version)"
fi

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
fi

echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

echo ""
echo "🎉 Launching Streamlit..."
streamlit run app.py