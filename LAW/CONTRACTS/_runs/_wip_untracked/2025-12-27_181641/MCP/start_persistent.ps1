# Start AGS MCP Server in a persistent background process
# This keeps the server running even when you close the terminal

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Restart
)

$REPO_ROOT = Split-Path -Parent $PSScriptRoot
$MCP_SERVER = Join-Path $REPO_ROOT "MCP\server.py"
$PID_FILE = Join-Path $REPO_ROOT "CONTRACTS\_runs\mcp_logs\server.pid"
$LOG_DIR = Join-Path $REPO_ROOT "CONTRACTS\_runs\mcp_logs"

# Ensure log directory exists
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

function Get-ServerStatus {
    if (Test-Path $PID_FILE) {
        $pid = Get-Content $PID_FILE
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "[RUNNING] AGS MCP Server (PID: $pid)" -ForegroundColor Green
            Write-Host "  Started: $($process.StartTime)" -ForegroundColor Cyan
            Write-Host "  CPU Time: $($process.CPU)" -ForegroundColor Cyan
            return $true
        } else {
            Write-Host "[STOPPED] PID file exists but process not found" -ForegroundColor Yellow
            Remove-Item $PID_FILE -Force
            return $false
        }
    } else {
        Write-Host "[STOPPED] AGS MCP Server is not running" -ForegroundColor Red
        return $false
    }
}

function Stop-Server {
    if (Test-Path $PID_FILE) {
        $pid = Get-Content $PID_FILE
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping AGS MCP Server (PID: $pid)..." -ForegroundColor Yellow
            Stop-Process -Id $pid -Force
            Remove-Item $PID_FILE -Force
            Write-Host "[STOPPED] Server terminated" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Process not found, cleaning up PID file" -ForegroundColor Yellow
            Remove-Item $PID_FILE -Force
        }
    } else {
        Write-Host "[INFO] Server is not running" -ForegroundColor Yellow
    }
}

function Start-Server {
    # Check if already running
    if (Get-ServerStatus) {
        Write-Host "[ERROR] Server is already running. Use -Restart to restart." -ForegroundColor Red
        return
    }

    Write-Host "Starting AGS MCP Server..." -ForegroundColor Cyan

    # Start server as a background job
    $job = Start-Job -ScriptBlock {
        param($ServerPath, $RepoRoot, $LogDir)
        Set-Location $RepoRoot
        $env:PYTHONUNBUFFERED = "1"
        python $ServerPath 2>&1 | Tee-Object -FilePath "$LogDir\server.log" -Append
    } -ArgumentList $MCP_SERVER, $REPO_ROOT, $LOG_DIR

    # Wait a moment for the job to start
    Start-Sleep -Seconds 2

    # Get the actual Python process PID (not the PowerShell job)
    $pythonProcess = Get-Process python -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*server.py*" } |
        Select-Object -First 1

    if ($pythonProcess) {
        $pythonProcess.Id | Out-File $PID_FILE -Force
        Write-Host "[STARTED] AGS MCP Server (PID: $($pythonProcess.Id))" -ForegroundColor Green
        Write-Host "  Job ID: $($job.Id)" -ForegroundColor Cyan
        Write-Host "  Log: $LOG_DIR\server.log" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "To check status: .\start_persistent.ps1 -Status" -ForegroundColor Yellow
        Write-Host "To stop server: .\start_persistent.ps1 -Stop" -ForegroundColor Yellow
    } else {
        Write-Host "[ERROR] Failed to start server - check logs" -ForegroundColor Red
        Remove-Job $job -Force
    }
}

# Main logic
if ($Status) {
    Get-ServerStatus
} elseif ($Stop) {
    Stop-Server
} elseif ($Restart) {
    Stop-Server
    Start-Sleep -Seconds 2
    Start-Server
} else {
    Start-Server
}
