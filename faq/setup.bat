@echo off
echo ========================================
echo FAQ NLP System Setup
echo ========================================

echo.
echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [4/5] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo.
echo [5/5] Setup completed successfully!
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo 1. Create MySQL database 'faq_nlp_system'
echo 2. Update database credentials in .env file
echo 3. Run: python database/init_db.py
echo 4. Run: python app.py
echo ========================================
echo.
pause
