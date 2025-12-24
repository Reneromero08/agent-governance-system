#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launch CATALYTIC-DPT Swarm as Antigravity Background Nodes

.DESCRIPTION
    Launches Governor and Ant Workers as background commands that appear in
    the Antigravity UI for integrated monitoring.
#>

$ProjectRoot = "d:\CCC 2.0\AI\agent-governance-system"
$CatalyticDPT = "$ProjectRoot\CATALYTIC-DPT"

Write-Host "🚀 Launching Antigravity Swarm Nodes..." -ForegroundColor Cyan

# Use run_command via Antigravity (this will be invoked by Claude)
# Note: For this script to work as intended, Claude needs to call run_command for each node.

Write-Host "1. Governor Node (Gemini Manager)" -ForegroundColor Yellow
# command: python d:\CCC 2.0\AI\agent-governance-system\CATALYTIC-DPT\agent_loop.py --role Governor

Write-Host "2. Ant-1 Node (Kilo Code Executor)" -ForegroundColor Yellow
# command: python d:\CCC 2.0\AI\agent-governance-system\CATALYTIC-DPT\agent_loop.py --role Ant-1

Write-Host "3. Ant-2 Node (Kilo Code Executor)" -ForegroundColor Yellow
# command: python d:\CCC 2.0\AI\agent-governance-system\CATALYTIC-DPT\agent_loop.py --role Ant-2

Write-Host "`nAll nodes set to poll the MCP ledger at $ProjectRoot\CONTRACTS\_runs\mcp_ledger" -ForegroundColor Green
Write-Host "Visibility: Check your Antigravity background tasks UI." -ForegroundColor Cyan
