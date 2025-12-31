<#
.SYNOPSIS
    AGS MCP Server Automatic Startup Script

.DESCRIPTION
    Automatically starts the AGS MCP Server on Windows boot.
    Reads configuration from autostart_config.json and manages server lifecycle.

.NOTES
    - Runs as a Windows Task Scheduler task
    - Logs to LAW/CONTRACTS/_runs/mcp_logs/
    - Monitors server health and restarts if needed
#>

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$REPO_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$CONFIG_FILE = Join-Path $PSScriptRoot "autostart_config.json"
$LOG_DIR = Join-Path $REPO_ROOT "LAW\CONTRACTS\_runs\mcp_logs"
$PID_FILE = Join-Path $LOG_DIR "server.pid"
$STARTUP_LOG = Join-Path $LOG_DIR "autostart.log"

# Ensure log directory exists
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Add-Content -Path $STARTUP_LOG -Value $logMessage
    Write-Host $logMessage
}

function Test-IsAdmin {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Config {
    if (Test-Path $CONFIG_FILE) {
        return Get-Content $CONFIG_FILE | ConvertFrom-Json
    }
    Write-Log "Config file not found, using defaults" "WARN"
    return $null
}

function Get-ServerStatus {
    if (Test-Path $PID_FILE) {
        $processId = Get-Content $PID_FILE
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            return @{
                Running = $true
                PID = $processId
                StartTime = $process.StartTime
                CPU = $process.CPU
            }
        }
    }
    return @{ Running = $false }
}

function Stop-Server {
    Write-Log "Stopping AGS MCP Server..."
    $status = Get-ServerStatus
    if ($status.Running) {
        Write-Log "Killing process PID: $($status.PID)"
        Stop-Process -Id $status.PID -Force -ErrorAction SilentlyContinue
        Remove-Item $PID_FILE -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Write-Log "Server stopped" "INFO"
    } else {
        Write-Log "Server is not running" "WARN"
    }
}

function Start-Server {
    Write-Log "Starting AGS MCP Server..."

    $config = Get-Config
    $status = Get-ServerStatus

    if ($status.Running) {
        Write-Log "Server already running (PID: $($status.PID))" "WARN"
        return
    }

    # Rebuild Cortex if enabled
    if ($config.components.cortex_rebuild.enabled -and $config.components.cortex_rebuild.on_startup) {
        Write-Log "Rebuilding Cortex index..."
        $cortexScript = Join-Path $REPO_ROOT $config.components.cortex_rebuild.script
        & python -- "$cortexScript" 2>&1 | Add-Content -Path $STARTUP_LOG
    }

    # Start MCP Server
    if ($config.components.mcp_server.enabled) {
        $env:MCP_KEEPALIVE = "1"
        $serverScript = Join-Path $REPO_ROOT $config.components.mcp_server.script
        $serverArg = '"' + $serverScript + '"'
        Write-Log "Launching MCP server: $serverScript"

        # Start server in background
        $process = Start-Process -FilePath "python" `
                                  -ArgumentList @($serverArg) `
                                  -WorkingDirectory $REPO_ROOT `
                                  -WindowStyle Hidden `
                                  -PassThru `
                                  -RedirectStandardOutput (Join-Path $LOG_DIR "server_stdout.log") `
                                  -RedirectStandardError (Join-Path $LOG_DIR "server_stderr.log")

        # Save PID
        $process.Id | Out-File $PID_FILE -Force
        Write-Log "Server started (PID: $($process.Id))" "INFO"

        # Wait for server to initialize
        Start-Sleep -Seconds 3

        # Verify it's still running
        if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
            Write-Log "Server failed to start - check error logs" "ERROR"
            return
        }

        Write-Log "Server is running and healthy" "INFO"
    }
}

function Install-StartupTask {
    Write-Log "Installing Windows Task Scheduler task..."

    $taskName = "AGS_MCP_Server_Autostart"
    $scriptPath = Join-Path $PSScriptRoot "autostart.ps1"

    # Remove existing task if present
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

    # Create new task
    $action = New-ScheduledTaskAction `
        -Execute "PowerShell.exe" `
        -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`" -Start"

    $trigger = New-ScheduledTaskTrigger -AtStartup

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)

    $runLevel = "Limited"
    if (Test-IsAdmin) {
        $runLevel = "Highest"
    }

    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType Interactive `
        -RunLevel $runLevel

    $fallbackInstalled = $false
    try {
        Register-ScheduledTask `
            -TaskName $taskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description "Automatically start AGS MCP Server on system boot" `
            -ErrorAction Stop
    } catch {
        Write-Log "Task install failed: $($_.Exception.Message)" "ERROR"
        Write-Log "Falling back to schtasks (per-user logon trigger)." "WARN"
        $escapedScript = $scriptPath.Replace('"', '""')
        $taskCmd = "PowerShell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File `"`"$escapedScript`"`" -Start"
        $schtasksPath = Join-Path $env:SystemRoot "System32\\schtasks.exe"
        $schtasksExe = Join-Path $env:SystemRoot "System32\\schtasks.exe"
        $cmd = "`"$schtasksExe`" /Create /TN `"$taskName`" /TR `"$taskCmd`" /SC ONLOGON /RL LIMITED /F /RU `"$env:USERNAME`""
        $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -NoNewWindow -Wait -PassThru
        if ($proc.ExitCode -ne 0) {
            Write-Log "schtasks fallback failed with exit code $($proc.ExitCode)." "ERROR"
            Write-Log "Falling back to Startup folder shortcut." "WARN"
            try {
                $startupDir = [Environment]::GetFolderPath("Startup")
                $shortcutPath = Join-Path $startupDir "AGS_MCP_Server_Autostart.lnk"
                $wsh = New-Object -ComObject WScript.Shell
                $shortcut = $wsh.CreateShortcut($shortcutPath)
                $shortcut.TargetPath = "PowerShell.exe"
                $shortcut.Arguments = "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`" -Start"
                $shortcut.WorkingDirectory = $REPO_ROOT
                $shortcut.WindowStyle = 7
                $shortcut.Save()
                Write-Log "Startup shortcut created at $shortcutPath" "INFO"
                $fallbackInstalled = $true
            } catch {
                Write-Log "Startup shortcut fallback failed: $($_.Exception.Message)" "ERROR"
            }
        }
    }

    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if (-not $task -and -not $fallbackInstalled) {
        Write-Log "Task install did not persist; task not found after registration." "ERROR"
        return
    }

    if ($task) {
        Write-Log "Task installed successfully (RunLevel: $runLevel)" "INFO"
    } elseif ($fallbackInstalled) {
        Write-Log "Autostart enabled via Startup folder shortcut." "INFO"
    }
    Write-Host ""
    Write-Host "Task '$taskName' installed!" -ForegroundColor Green
    Write-Host "The MCP server will now start automatically on boot." -ForegroundColor Cyan
}

function Uninstall-StartupTask {
    Write-Log "Uninstalling Windows Task Scheduler task..."

    $taskName = "AGS_MCP_Server_Autostart"
    schtasks /Delete /TN $taskName /F

    Write-Log "Task uninstalled" "INFO"
    Write-Host "Task removed. Server will no longer start automatically." -ForegroundColor Yellow
}

function Show-Status {
    Write-Host ""
    Write-Host "=== AGS MCP Server Status ===" -ForegroundColor Cyan

    $status = Get-ServerStatus
    if ($status.Running) {
        Write-Host "Status: RUNNING" -ForegroundColor Green
        Write-Host "PID: $($status.PID)" -ForegroundColor Cyan
        Write-Host "Started: $($status.StartTime)" -ForegroundColor Cyan
        Write-Host "CPU Time: $([math]::Round($status.CPU, 2))s" -ForegroundColor Cyan
    } else {
        Write-Host "Status: STOPPED" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "Task Scheduler Status:" -ForegroundColor Cyan
    $task = Get-ScheduledTask -TaskName "AGS_MCP_Server_Autostart" -ErrorAction SilentlyContinue
    if ($task) {
        Write-Host "  Autostart: ENABLED ($($task.State))" -ForegroundColor Green
    } else {
        $startupDir = [Environment]::GetFolderPath("Startup")
        $shortcutPath = Join-Path $startupDir "AGS_MCP_Server_Autostart.lnk"
        if (Test-Path $shortcutPath) {
            Write-Host "  Autostart: ENABLED (Startup folder)" -ForegroundColor Green
        } else {
            Write-Host "  Autostart: DISABLED" -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Log Files:" -ForegroundColor Cyan
    Write-Host "  Startup: $STARTUP_LOG" -ForegroundColor Gray
    Write-Host "  Server: $LOG_DIR\server_stdout.log" -ForegroundColor Gray
    Write-Host ""
}

# Main logic
try {
    if ($Install) {
        Install-StartupTask
    } elseif ($Uninstall) {
        Uninstall-StartupTask
    } elseif ($Start) {
        Start-Server
    } elseif ($Stop) {
        Stop-Server
    } elseif ($Restart) {
        Stop-Server
        Start-Sleep -Seconds 2
        Start-Server
    } elseif ($Status) {
        Show-Status
    } else {
        Write-Host "AGS MCP Server Autostart Manager" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage:"
        Write-Host "  .\autostart.ps1 -Install     Install autostart (runs on boot)"
        Write-Host "  .\autostart.ps1 -Uninstall   Remove autostart"
        Write-Host "  .\autostart.ps1 -Start       Start server now"
        Write-Host "  .\autostart.ps1 -Stop        Stop server"
        Write-Host "  .\autostart.ps1 -Restart     Restart server"
        Write-Host "  .\autostart.ps1 -Status      Show server status"
        Write-Host ""
    }
} catch {
    Write-Log "ERROR: $_" "ERROR"
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
