@echo off

REM HRHUB Quick Start Script for Windows

echo 🚀 Starting HRHUB...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -q -r requirements.txt
echo ✅ Dependencies installed

echo.
echo 🎉 Launching Streamlit app...
echo 📍 Open your browser to: http://localhost:8501
echo.

streamlit run app.py
