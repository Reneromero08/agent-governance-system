#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launch CATALYTIC-DPT Swarm - multiple modes supported.

.DESCRIPTION
    Mode 1 (default): Python polling agents (no external CLI needed)
    Mode 2: VSCode terminals via Antigravity Bridge (port 4000)
    Mode 3: Direct gemini/kilocode CLI launch

.EXAMPLE
    .\launch_swarm.ps1              # Python polling mode
    .\launch_swarm.ps1 -Mode bridge # VSCode terminal mode
    .\launch_swarm.ps1 -Mode cli    # Direct CLI mode
#>

param(
    [ValidateSet("python", "bridge", "cli")]
    [string]$Mode = "python"
)

$ProjectRoot = "d:\CCC 2.0\AI\agent-governance-system"
$CatalyticDPT = "$ProjectRoot\CATALYTIC-DPT"
$BridgeUrl = "http://127.0.0.1:4000/terminal"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CATALYTIC-DPT Swarm Launcher" -ForegroundColor Cyan
Write-Host "  Mode: $Mode" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan

# Initialize MCP ledger directory
$LedgerPath = "$ProjectRoot\CONTRACTS\_runs\mcp_ledger"
if (!(Test-Path $LedgerPath)) {
    New-Item -ItemType Directory -Path $LedgerPath -Force | Out-Null
    Write-Host "[OK] Created ledger directory" -ForegroundColor Green
}

function Launch-Bridge {
    param($Name, $Command)
    $Payload = @{
        name           = $Name
        cwd            = $CatalyticDPT
        initialCommand = $Command
    } | ConvertTo-Json

    Write-Host "Launching $Name via Bridge..." -NoNewline
    try {
        Invoke-RestMethod -Uri $BridgeUrl -Method Post -Body $Payload -ContentType "application/json" -TimeoutSec 2 | Out-Null
        Write-Host " [OK]" -ForegroundColor Green
    }
    catch {
        Write-Host " [FAIL] Bridge not available on port 4000" -ForegroundColor Red
    }
}

switch ($Mode) {
    "python" {
        Write-Host "`nStarting Python polling agents..." -ForegroundColor Yellow
        Write-Host "Open 3 terminals and run:" -ForegroundColor White
        Write-Host ""
        Write-Host "  Terminal 1 (Governor):" -ForegroundColor Cyan
        Write-Host "    cd `"$CatalyticDPT`"" -ForegroundColor Gray
        Write-Host "    python SKILLS/swarm-orchestrator/poll_and_execute.py --role Governor" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Terminal 2 (Ant-1):" -ForegroundColor Cyan
        Write-Host "    cd `"$CatalyticDPT`"" -ForegroundColor Gray
        Write-Host "    python SKILLS/swarm-orchestrator/poll_and_execute.py --role Ant-1" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Terminal 3 (Ant-2):" -ForegroundColor Cyan
        Write-Host "    cd `"$CatalyticDPT`"" -ForegroundColor Gray
        Write-Host "    python SKILLS/swarm-orchestrator/poll_and_execute.py --role Ant-2" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Or start Governor now? (y/n): " -NoNewline -ForegroundColor Yellow
        $answer = Read-Host
        if ($answer -eq "y") {
            Set-Location $CatalyticDPT
            python SKILLS/swarm-orchestrator/poll_and_execute.py --role Governor
        }
    }
    "bridge" {
        Write-Host "`nLaunching via Antigravity Bridge (port 4000)..." -ForegroundColor Yellow
        Launch-Bridge -Name "Governor" -Command "python SKILLS/swarm-orchestrator/poll_and_execute.py --role Governor"
        Launch-Bridge -Name "Ant-1" -Command "python SKILLS/swarm-orchestrator/poll_and_execute.py --role Ant-1"
        Launch-Bridge -Name "Ant-2" -Command "python SKILLS/swarm-orchestrator/poll_and_execute.py --role Ant-2"
        Write-Host "`nSwarm launched in VSCode panel!" -ForegroundColor Green
    }
    "cli" {
        Write-Host "`nLaunching CLI agents (requires gemini + kilocode installed)..." -ForegroundColor Yellow
        Write-Host "Starting Governor with Gemini CLI..." -ForegroundColor Cyan

        # Check if gemini exists
        $gemini = Get-Command gemini -ErrorAction SilentlyContinue
        if ($gemini) {
            Start-Process -FilePath "cmd" -ArgumentList "/k cd /d `"$CatalyticDPT`" && gemini"
            Write-Host "[OK] Gemini launched" -ForegroundColor Green
        } else {
            Write-Host "[SKIP] gemini not found in PATH" -ForegroundColor Yellow
        }

        # Launch Ant workers with npx
        $npx = Get-Command npx -ErrorAction SilentlyContinue
        if ($npx) {
            $AntPrompt = "You are ANT-1. Poll MCP for tasks via get_pending_tasks. Execute file operations exactly. Report results."
            Start-Process -FilePath "cmd" -ArgumentList "/k cd /d `"$CatalyticDPT`" && npx @kilocode/cli `"$AntPrompt`""
            Write-Host "[OK] Ant-1 launched" -ForegroundColor Green
        } else {
            Write-Host "[SKIP] npx not found" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n============================================" -ForegroundColor Cyan
