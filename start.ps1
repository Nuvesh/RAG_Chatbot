# RAG Studio - Quick Start Script
# Run this to start the server

Write-Host "🚀 Starting RAG Studio..." -ForegroundColor Cyan
Write-Host ""

# Check if GEMINI_API_KEY is set
if ([string]::IsNullOrEmpty($env:GEMINI_API_KEY)) {
    Write-Host "⚠️  Warning: GEMINI_API_KEY is not set!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can:" -ForegroundColor Gray
    Write-Host "  1. Set it now: `$env:GEMINI_API_KEY='your-key-here'" -ForegroundColor Gray
    Write-Host "  2. Start without it and set later in the UI" -ForegroundColor Gray
    Write-Host ""
    
    $response = Read-Host "Do you want to set the API key now? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        $apiKey = Read-Host "Enter your Gemini API key"
        $env:GEMINI_API_KEY = $apiKey
        Write-Host "✅ API key set for this session" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  You can set it later or use the UI to input the key" -ForegroundColor Cyan
    }
} else {
    Write-Host "✅ GEMINI_API_KEY found" -ForegroundColor Green
}

Write-Host ""
Write-Host "📁 Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "🌐 Starting server at http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

uvicorn main:app --reload
