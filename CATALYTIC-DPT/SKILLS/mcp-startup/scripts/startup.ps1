# MCP Network Startup Script (PowerShell)
# Usage:
#   .\startup.ps1                    # Interactive mode
#   .\startup.ps1 -All               # Start everything
#   .\startup.ps1 -OllamaOnly       # Start only Ollama
#   .\startup.ps1 -MCPOnly          # Start only MCP server
#   .\startup.ps1 -Ants 2           # Start with 2 Ant workers

param(
    [switch]$All,
    [switch]$OllamaOnly,
    [switch]$MCPOnly,
    [switch]$Interactive,
    [int]$Ants = 2
)

# Get repo root
$repoRoot = (Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.FullName
Set-Location $repoRoot

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "MCP NETWORK STARTUP (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Interactive mode
if ($Interactive -or (-not $All -and -not $OllamaOnly -and -not $MCPOnly)) {
    Write-Host "What would you like to start?" -ForegroundColor Yellow
    Write-Host "1) Full network (Ollama + MCP + Ants)"
    Write-Host "2) Ollama only"
    Write-Host "3) MCP server only"
    Write-Host "4) Exit"
    Write-Host ""
    $choice = Read-Host "Enter choice (1-4)"

    switch ($choice) {
        "1" { $All = $true }
        "2" { $OllamaOnly = $true }
        "3" { $MCPOnly = $true }
        "4" { Write-Host "Exiting" -ForegroundColor Yellow; exit }
        default { Write-Host "Invalid choice" -ForegroundColor Red; exit 1 }
    }
}

# Start Ollama if needed
if ($All -or $OllamaOnly) {
    Write-Host "Starting Ollama server..." -ForegroundColor Cyan

    # Check if already running
    $ollama = Test-Path "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
    if ($ollama) {
        # Start Ollama in background
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Minimized
        Write-Host "Ollama process launched" -ForegroundColor Green

        # Wait for it to be ready
        Write-Host "Waiting for Ollama to be ready..." -ForegroundColor Yellow
        $timeout = 0
        while ($timeout -lt 30) {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -ErrorAction SilentlyContinue
                if ($response.StatusCode -eq 200) {
                    Write-Host "Ollama is online" -ForegroundColor Green
                    break
                }
            } catch {}

            Start-Sleep -Seconds 2
            $timeout += 2
        }
    } else {
        Write-Host "Ollama not found. Install from https://ollama.ai/" -ForegroundColor Red
        exit 1
    }
}

# Start MCP server
if ($All -or $MCPOnly) {
    Write-Host ""
    Write-Host "Starting MCP server..." -ForegroundColor Cyan

    $mcpScript = "CATALYTIC-DPT\LAB\MCP\stdio_server.py"
    if (Test-Path $mcpScript) {
        Write-Host ""
        Write-Host "Launch the MCP server in a new terminal with:" -ForegroundColor Yellow
        Write-Host "  python $mcpScript" -ForegroundColor Yellow
        Write-Host ""

        # Optional: Auto-launch in new terminal
        $launch = Read-Host "Auto-launch in new terminal? (y/n)"
        if ($launch -eq 'y') {
            Start-Process -FilePath "python" -ArgumentList "$mcpScript" -NoNewWindow
            Write-Host "MCP server starting..." -ForegroundColor Green
        }
    } else {
        Write-Host "MCP server script not found: $mcpScript" -ForegroundColor Red
        exit 1
    }
}

# Start Ant workers
if ($All) {
    Write-Host ""
    Write-Host "Starting $Ants Ant workers..." -ForegroundColor Cyan

    $antScript = "CATALYTIC-DPT\SKILLS\ant-worker\scripts\ant_agent.py"
    if (Test-Path $antScript) {
        for ($i = 1; $i -le $Ants; $i++) {
            $agentId = "Ant-$i"
            Write-Host "Starting $agentId..." -ForegroundColor Cyan

            # Start in new terminal window
            Start-Process -FilePath "python" -ArgumentList "$antScript --agent_id $agentId"
        }
        Write-Host "Ant workers launched" -ForegroundColor Green
    } else {
        Write-Host "Ant worker script not found: $antScript" -ForegroundColor Red
    }
}

# Health check
Write-Host ""
Write-Host "=== Health Check ===" -ForegroundColor Cyan
Write-Host ""

try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Ollama server is running" -ForegroundColor Green

        $models = ($response.Content | ConvertFrom-Json).models
        if ($models.Count -gt 0) {
            Write-Host "[OK] LFM2 model is loaded" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] No models loaded" -ForegroundColor Red
        }
    } else {
        Write-Host "[FAIL] Ollama server is NOT running" -ForegroundColor Red
    }
} catch {
    Write-Host "[FAIL] Ollama server is NOT running" -ForegroundColor Red
}

if (Test-Path "CONTRACTS\_runs\mcp_ledger") {
    Write-Host "[OK] MCP ledger exists" -ForegroundColor Green
} else {
    Write-Host "[FAIL] MCP ledger does NOT exist" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Connect Claude Desktop to your MCP server"
Write-Host "  2. Send tasks from Claude to your Ant workers"
Write-Host "  3. Monitor output in the MCP and Ant terminals"
Write-Host "================================================" -ForegroundColor Cyan
