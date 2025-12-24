#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launch CATALYTIC-DPT Swarm in VSCode terminals via Antigravity Bridge.
#>

$ProjectRoot = "d:\CCC 2.0\AI\agent-governance-system"
$CatalyticDPT = "$ProjectRoot\CATALYTIC-DPT"
$BridgeUrl = "http://127.0.0.1:4000/terminal"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CATALYTIC-DPT Swarm Launcher (Bridge)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

function Launch-Terminal {
    param($Name, $Command)
    
    $Payload = @{
        name           = $Name
        cwd            = $CatalyticDPT
        initialCommand = $Command
    } | ConvertTo-Json
    
    Write-Host "Launching $Name..." -NoNewline
    try {
        $Response = Invoke-RestMethod -Uri $BridgeUrl -Method Post -Body $Payload -ContentType "application/json" -TimeoutSec 2
        Write-Host " ✅" -ForegroundColor Green
    }
    catch {
        Write-Host " ❌ (Bridge connection failed. Is port 4000 open?)" -ForegroundColor Red
    }
}

# 1. Initialize MCP ledger
$LedgerPath = "$ProjectRoot\CONTRACTS\_runs\mcp_ledger"
if (!(Test-Path $LedgerPath)) { New-Item -ItemType Directory -Path $LedgerPath -Force | Out-Null }

# 2. Launch agents
Launch-Terminal -Name "Governor" -Command "gemini"

$AntPrompt = "You are an ANT WORKER. Poll for tasks: get_pending_tasks. Execute templates EXACTLY. Report result. NEVER deviate."
Launch-Terminal -Name "Ant-1" -Command "npx @kilocode/cli `"$AntPrompt`""
Launch-Terminal -Name "Ant-2" -Command "npx @kilocode/cli `"$AntPrompt`""

Write-Host ""
Write-Host "Swarm launched in VSCode panel!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
