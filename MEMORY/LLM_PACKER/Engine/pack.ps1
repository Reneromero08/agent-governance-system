param(
  [ValidateSet("ags", "lab")]
  [string]$Scope = "ags",
  [string]$OutDir = "",
  [ValidateSet("full", "delta")]
  [string]$Mode = "full",
  [ValidateSet("full", "lite")]
  [string]$Profile = "full",
  [string]$Stamp = "",
  [switch]$Zip,
  [switch]$NoZip,
  [switch]$Combined,
  [switch]$NoCombined,
  [switch]$SplitLite
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  $packerParentDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
  return (Resolve-Path (Split-Path -Parent $packerParentDir)).Path
}

$repoRoot = Get-RepoRoot
$pythonModule = "MEMORY.LLM_PACKER.Engine.packer"

if ($Stamp -eq "") { $Stamp = (Get-Date).ToString("yyyy-MM-dd_HH-mm-ss") }

$envProfile = $env:PACK_PROFILE
if ($Profile -eq "full" -and -not [string]::IsNullOrWhiteSpace($envProfile)) {
  $envProfileLower = $envProfile.Trim().ToLowerInvariant()
  if ($envProfileLower -in @("full", "lite")) {
    $Profile = $envProfileLower
  }
}

$zipEnabled = $true
$combinedEnabled = $true
$splitLiteEnabled = $false

if ($Profile -eq "lite") {
  $zipEnabled = $false
  $combinedEnabled = $false
  $splitLiteEnabled = $true
}

if ($PSBoundParameters.ContainsKey("Zip")) { $zipEnabled = $true }
if ($PSBoundParameters.ContainsKey("NoZip")) { $zipEnabled = $false }

if ($PSBoundParameters.ContainsKey("Combined")) { $combinedEnabled = $true }
if ($PSBoundParameters.ContainsKey("NoCombined")) { $combinedEnabled = $false }

if ($PSBoundParameters.ContainsKey("SplitLite")) { $splitLiteEnabled = $true }

if ($OutDir -eq "") {
  if ($Scope -eq "lab") {
    $OutDir = "MEMORY/LLM_PACKER/_packs/lab-pack-$Stamp"
  } else {
    $OutDir = "MEMORY/LLM_PACKER/_packs/ags-pack-$Stamp"
  }
}

$args = @(
  "python",
  "-m",
  $pythonModule,
  "--scope", $Scope,
  "--mode", $Mode,
  "--profile", $Profile,
  "--out-dir", $OutDir
)

$args += @("--stamp", $Stamp)
if ($zipEnabled) { $args += "--zip" }
if ($combinedEnabled) { $args += "--combined" }
if ($splitLiteEnabled) { $args += "--split-lite" }

Write-Host "Running: $($args -join ' ')"
Push-Location $repoRoot
try {
  & $args[0] $args[1..($args.Count - 1)]
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Pop-Location
}
