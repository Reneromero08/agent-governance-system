#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launch CATALYTIC-DPT Swarm: Governor + 2 Ant Workers

.DESCRIPTION
    Starts the MCP server, Governor (Gemini CLI), and 2 Ant Workers (Kilo Code)
    in separate terminal windows for parallel execution.

.EXAMPLE
    .\launch_swarm.ps1
#>

$ErrorActionPreference = "Continue"
$ProjectRoot = "d:\CCC 2.0\AI\agent-governance-system"
$CatalyticDPT = "$ProjectRoot\CATALYTIC-DPT"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CATALYTIC-DPT Swarm Launcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Initialize MCP ledger
Write-Host "[1/4] Initializing MCP ledger..." -ForegroundColor Yellow
$LedgerPath = "$ProjectRoot\CONTRACTS\_runs\mcp_ledger"
New-Item -ItemType Directory -Path $LedgerPath -Force | Out-Null
Write-Host "      Ledger: $LedgerPath" -ForegroundColor Green

# 2. Start Governor (Gemini CLI)
Write-Host "[2/4] Starting Governor (Gemini CLI)..." -ForegroundColor Yellow

$GovernorPrompt = @"
You are the GOVERNOR in the CATALYTIC-DPT swarm system.

CHAIN OF COMMAND:
- You receive tasks from Claude (above you)
- You dispatch tasks to Ant Workers (below you)
- Escalate issues UP to Claude when uncertain

YOUR ROLE:
1. Break high-level goals into ant-sized subtasks
2. Create strict task templates for Ants
3. Dispatch via MCP: dispatch_task(to_agent='Ant-1', task_spec={...})
4. Monitor results via get_results()
5. Aggregate and report back

CURRENT DIRECTORY: $CatalyticDPT
SKILLS AVAILABLE: governor/, ant-worker/, file-analyzer/
MCP SERVER: $CatalyticDPT\MCP\server.py

Await instructions from Claude.
"@

# Launch Governor in new terminal
Start-Process wt -ArgumentList "new-tab", "--title", "Governor", "pwsh", "-NoExit", "-Command", "cd '$CatalyticDPT'; Write-Host 'Governor Ready' -ForegroundColor Green; gemini"
Write-Host "      Governor terminal opened" -ForegroundColor Green

# 3. Start Ant Workers
Write-Host "[3/4] Starting 2 Ant Workers (Kilo Code)..." -ForegroundColor Yellow

$AntPrompt = @"
You are an ANT WORKER in the CATALYTIC-DPT swarm.

CHAIN OF COMMAND:
- You receive tasks from Governor (above you)
- You execute templates EXACTLY as given
- Escalate to Governor if uncertain

YOUR ROLE:
1. Poll for tasks: get_pending_tasks('Ant-{N}')
2. Execute the task template (fill placeholders only)
3. Report result: report_result(task_id, status, result)
4. NEVER deviate from template structure

RULES:
- Schema-valid output or HARD FAIL
- Do NOT improvise - escalate instead
- Log everything to MCP ledger

CURRENT DIRECTORY: $CatalyticDPT
"@

# Launch Ant-1
Start-Process wt -ArgumentList "new-tab", "--title", "Ant-1", "pwsh", "-NoExit", "-Command", "cd '$CatalyticDPT'; Write-Host 'Ant-1 Ready' -ForegroundColor Yellow; npx @kilocode/cli"
Write-Host "      Ant-1 terminal opened" -ForegroundColor Green

# Launch Ant-2
Start-Process wt -ArgumentList "new-tab", "--title", "Ant-2", "pwsh", "-NoExit", "-Command", "cd '$CatalyticDPT'; Write-Host 'Ant-2 Ready' -ForegroundColor Yellow; npx @kilocode/cli"
Write-Host "      Ant-2 terminal opened" -ForegroundColor Green

# 4. Summary
Write-Host ""
Write-Host "[4/4] Swarm launched!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Governor:  Gemini CLI (Tab 1)" -ForegroundColor White
Write-Host "  Ant-1:     Kilo Code (Tab 2)" -ForegroundColor White
Write-Host "  Ant-2:     Kilo Code (Tab 3)" -ForegroundColor White
Write-Host "  Ledger:    $LedgerPath" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Claude can now dispatch to Governor via:" -ForegroundColor Cyan
Write-Host "  mcp_server.send_directive('Claude', 'Governor', 'Your task here', {})" -ForegroundColor White
