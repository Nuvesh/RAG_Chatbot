@echo off
echo ========================================
echo   RAG Studio - PDF Q&A System
echo ========================================
echo.

REM Check if GEMINI_API_KEY is set
if "%GEMINI_API_KEY%"=="" (
    echo [WARNING] GEMINI_API_KEY is not set!
    echo.
    echo Please set your Gemini API key first:
    echo set GEMINI_API_KEY=your-api-key-here
    echo.
    set /p "SET_KEY=Do you want to set it now? (y/n): "
    if /i "%SET_KEY%"=="y" (
        set /p "GEMINI_API_KEY=Enter your Gemini API key: "
        echo.
        echo API key set for this session
    ) else (
        echo.
        echo Starting without API key - you can set it in the UI later
    )
) else (
    echo [OK] GEMINI_API_KEY found
)

echo.
echo ========================================
echo   Installing dependencies...
echo ========================================
pip install -r requirements.txt --quiet

echo.
echo ========================================
echo   Starting RAG Studio Server...
echo ========================================
echo.
echo Open your browser to: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

uvicorn main:app --reload
