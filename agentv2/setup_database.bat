@echo off
echo ===============================================
echo AgentV2 - Database Setup
echo ===============================================
echo.

echo Please make sure MySQL is running before continuing...
echo.
pause

echo Creating AgentV2 database...
echo Please enter your MySQL root password when prompted.
echo.

echo Step 1: Creating database...
mysql -u root -p < database\create_database.sql
if %errorlevel% neq 0 (
    echo ERROR: Failed to create database!
    echo Please check if MySQL is running and credentials are correct.
    pause
    exit /b 1
)

echo.
echo Step 2: Creating tables and schema...
mysql -u root -p agentv2_db < database\schema.sql
if %errorlevel% neq 0 (
    echo ERROR: Failed to create schema!
    echo Please check the schema.sql file for errors.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo ✅ Database setup completed successfully!
echo ===============================================
echo.
echo Database: agentv2_db
echo Tables created: damage_classes, user_reports, yolo_detections, 
echo                 risk_analysis, repair_procedures, maintenance_schedule,
echo                 validation_actions, trainer_data, training_sessions,
echo                 model_metrics, model_calibration, system_config
echo.
echo Views created: dashboard_summary, model_performance, 
echo                risk_distribution, damage_frequency, maintenance_efficiency
echo.
echo Next step: Edit .env file with database credentials, then run setup.bat
echo.
pause
