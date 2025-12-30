while ($true) {
    Clear-Host
    Write-Host "🐜 SWARM MONITOR 🐜" -ForegroundColor Green
    Write-Host "===================" -ForegroundColor Green
    
    # Show active tasks
    $active = Get-ChildItem "INBOX/agents/Local Models/ACTIVE_TASKS/*.json" | Measure-Object | Select-Object -ExpandProperty Count
    Write-Host "Active Agents: $active" -ForegroundColor Cyan
    
    Write-Host "`n📜 LATEST LOGS:" -ForegroundColor Yellow
    Get-Content swarm_debug.log -Tail 20
    
    Start-Sleep -Seconds 2
}
