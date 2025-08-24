@echo off
echo ===============================================
echo AgentV2 - System Validation
echo ===============================================
echo.

echo Activating virtual environment...
if exist "agentv2_env\Scripts\activate.bat" (
    call agentv2_env\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Running system validation tests...
echo This will check all components and functionality.
echo.

python validate_system.py

echo.
if %errorlevel% equ 0 (
    echo ===============================================
    echo ✅ System validation completed successfully!
    echo ===============================================
    echo.
    echo Your AgentV2 system is ready to run.
    echo Next step: run.bat
) else (
    echo ===============================================
    echo ❌ System validation failed!
    echo ===============================================
    echo.
    echo Please check the errors above and fix them.
    echo See validation_results.json for detailed results.
)

echo.
pause
