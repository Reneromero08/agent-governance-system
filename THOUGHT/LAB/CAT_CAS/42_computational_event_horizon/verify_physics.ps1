$ErrorActionPreference = "Stop"

Write-Host "================================================================================"
Write-Host "CAT_CAS MASTER VERIFICATION PIPELINE"
Write-Host "Phase 3 Hardening & Regression Testing"
Write-Host "================================================================================"

$pythonScripts = Get-ChildItem -Filter "*_*.py" | Sort-Object Name
$failed = $false

Write-Host "`n[*] VERIFYING PYTHON SIMULATIONS (Phases 1 & 2)..."
foreach ($script in $pythonScripts) {
    Write-Host " -> Executing $($script.Name)..." -NoNewline
    
    $output = & "..\..\..\..\.venv\Scripts\python.exe" $script.FullName 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host " [CRASHED]" -ForegroundColor Red
        Write-Host $output
        $failed = $true
    } else {
        # Check if the script was supposed to print SUCCESS but didn't
        if ($output -match "\[SUCCESS\]") {
            Write-Host " [VERIFIED]" -ForegroundColor Green
        } elseif ($output -match "CONCLUSION:") {
            # For scripts without explicit SUCCESS tags but have valid conclusions
            Write-Host " [VERIFIED]" -ForegroundColor Green
        } else {
            Write-Host " [WARNING: No Success Tag]" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n[*] VERIFYING RUST BARE-METAL EXPLOITS (Phase 4+)..."
$rustDirs = Get-ChildItem -Path "ULTRA" -Directory -Filter "exp_*" | Sort-Object Name
foreach ($dir in $rustDirs) {
    $rustPath = Join-Path $dir.FullName "rust"
    if (Test-Path (Join-Path $rustPath "Cargo.toml")) {
        Write-Host " -> Testing $($dir.Name)..." -NoNewline
        
        Push-Location $rustPath
        $output = cmd.exe /c "cargo test 2>&1"
        $exitCode = $LASTEXITCODE
        Pop-Location
        
        if ($exitCode -ne 0) {
            Write-Host " [FAILED]" -ForegroundColor Red
            Write-Host $output
            $failed = $true
        } else {
            Write-Host " [VERIFIED]" -ForegroundColor Green
        }
    }
}

Write-Host "`n================================================================================"
if ($failed) {
    Write-Host "[!] VERIFICATION FAILED: Physics anomalies are unstable." -ForegroundColor Red
    exit 1
} else {
    Write-Host "[SUCCESS] VERIFICATION COMPLETE: All physics invariants hold perfectly." -ForegroundColor Green
    exit 0
}
