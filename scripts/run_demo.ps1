<#
.SYNOPSIS
    VidGen one-click local demo script.

.DESCRIPTION
    Starts the API server and submits a demo generation job.
    Press Ctrl+C to stop the API server when done.

.EXAMPLE
    .\scripts\run_demo.ps1
    .\scripts\run_demo.ps1 -Port 8001
    .\scripts\run_demo.ps1 -Prompt "A cat sitting on a windowsill at sunset"
#>
param(
    [int]$Port = 8000,
    [string]$Prompt = "A robot walking forward in a city street at night",
    [int]$Seed = 101,
    [string]$Model = "stable-video-diffusion-img2vid"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "=== VidGen Local Demo ===" -ForegroundColor Cyan
Write-Host ""

# Check .env configuration
$envContent = Get-Content ".env" -Encoding UTF8 -ErrorAction SilentlyContinue
$mockSetting = ($envContent | Where-Object { $_ -match "^VIDGEN_USE_MOCK_PIPELINE=" }) -replace "VIDGEN_USE_MOCK_PIPELINE=", ""
if ($mockSetting -eq "true") {
    Write-Host "[WARN] VIDGEN_USE_MOCK_PIPELINE=true in .env" -ForegroundColor Yellow
    Write-Host "       For real model inference, set VIDGEN_USE_MOCK_PIPELINE=false" -ForegroundColor Yellow
    Write-Host ""
}

# Start API server in background
Write-Host "[1/3] Starting API server on port $Port ..." -ForegroundColor Green
$serverJob = Start-Job -ScriptBlock {
    param($Root, $PortNum)
    Set-Location $Root
    conda activate VidGen 2>$null
    uvicorn app.main:app --host 0.0.0.0 --port $PortNum 2>&1
} -ArgumentList $ProjectRoot, $Port

# Wait for server to be ready
Write-Host "[2/3] Waiting for API server ..." -ForegroundColor Green
$baseUrl = "http://127.0.0.1:$Port"
$maxWait = 30
$waited = 0
while ($waited -lt $maxWait) {
    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 2 -ErrorAction Stop
        Write-Host "       Server ready. mock_pipeline=$($response.mock_pipeline)" -ForegroundColor Gray
        break
    } catch {
        Start-Sleep -Seconds 2
        $waited += 2
    }
}
if ($waited -ge $maxWait) {
    Write-Host "[ERROR] Server did not start within $maxWait seconds" -ForegroundColor Red
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -ErrorAction SilentlyContinue
    exit 1
}

# Submit demo job
Write-Host "[3/3] Submitting demo job ..." -ForegroundColor Green
Write-Host "       Prompt: $Prompt" -ForegroundColor Gray
Write-Host "       Model:  $Model | Seed: $Seed" -ForegroundColor Gray
Write-Host ""

python scripts/submit_demo.py --base-url $baseUrl --prompt $Prompt --model $Model --seed $Seed --poll-seconds 5

Write-Host ""
Write-Host "=== Demo complete. Press Ctrl+C to stop the API server ===" -ForegroundColor Cyan
Write-Host "       Or run:  Stop-Job $serverJob; Remove-Job $serverJob" -ForegroundColor Gray

# Keep server running until user cancels
try {
    while ($true) { Start-Sleep -Seconds 1 }
} finally {
    Write-Host "Stopping API server ..." -ForegroundColor Yellow
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -ErrorAction SilentlyContinue
}
