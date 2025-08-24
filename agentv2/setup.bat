@echo off
echo ===============================================
echo AgentV2 - Advanced AI Asset Inspection System
echo ===============================================
echo.

echo Creating project directories...
mkdir uploads\user_reports 2>nul
mkdir uploads\training\images 2>nul
mkdir uploads\training\annotations 2>nul
mkdir models 2>nul
mkdir logs 2>nul
mkdir temp 2>nul
mkdir backups 2>nul

echo.
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.9+ first.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv agentv2_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call agentv2_env\Scripts\activate.bat

echo.
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo Checking MySQL connection...
python -c "import pymysql; print('PyMySQL installed successfully')"
if %errorlevel% neq 0 (
    echo WARNING: MySQL connection may not work properly
)

echo.
echo Setting up configuration...
if not exist ".env" (
    echo Creating .env file...
    echo # AgentV2 Environment Configuration > .env
    echo DB_HOST=localhost >> .env
    echo DB_PORT=3306 >> .env
    echo DB_USER=root >> .env
    echo DB_PASSWORD= >> .env
    echo DB_NAME=agentv2_db >> .env
    echo DEBUG=true >> .env
    echo API_HOST=0.0.0.0 >> .env
    echo API_PORT=8000 >> .env
    echo SECRET_KEY=agentv2-super-secret-key-change-in-production >> .env
    echo YOLO_DEVICE=cpu >> .env
    echo LOG_LEVEL=INFO >> .env
    echo.
    echo .env file created! Please edit it with your database credentials.
)

echo.
echo Downloading YOLO model...
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO model downloaded successfully')"
if %errorlevel% neq 0 (
    echo WARNING: YOLO model download failed. It will be downloaded on first run.
)

echo.
echo ===============================================
echo Setup completed successfully!
echo ===============================================
echo.
echo Next steps:
echo 1. Edit .env file with your database credentials
echo 2. Create database: CREATE DATABASE agentv2_db;
echo 3. Import schema: mysql -u root -p agentv2_db ^< database\schema.sql
echo 4. Run the application: run.bat
echo.
echo For detailed instructions, see README.md
echo.
pause
