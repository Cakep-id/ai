@echo off
echo ========================================
echo Starting FAQ NLP System
echo ========================================

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Testing database connection...
python -c "from database.connection import db_manager; print('Database:', 'OK' if db_manager.test_connection() else 'FAILED')"

echo.
echo Starting Flask server...
echo.
echo Open your browser and go to: http://127.0.0.1:5000
echo Press Ctrl+C to stop the server
echo.
python app.py
