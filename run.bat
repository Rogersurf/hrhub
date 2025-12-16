@echo off

echo 🚀 Starting HRHUB...
echo.

REM Use Python Launcher if available, otherwise regular python
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.10 -m venv venv 2>nul || py -3 -m venv venv 2>nul || python -m venv venv
) else (
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo 🎉 Launching Streamlit...
streamlit run app.py