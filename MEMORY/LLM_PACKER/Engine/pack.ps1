param(
  [ValidateSet("ags", "catalytic-dpt", "catalytic-dpt-lab")]
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
$packer = Join-Path $PSScriptRoot "packer.py"

if (-not (Test-Path -LiteralPath $packer)) {
  throw "Missing Python packer at: $packer"
}

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
  if ($Scope -eq "catalytic-dpt") {
    $OutDir = "MEMORY/LLM_PACKER/_packs/catalytic-dpt-pack-$Stamp"
  } elseif ($Profile -eq "lite") {
    $OutDir = "MEMORY/LLM_PACKER/_packs/llm-pack-lite-$Stamp"
  } else {
    $OutDir = "MEMORY/LLM_PACKER/_packs/llm-pack-$Stamp"
  }
}

$args = @(
  "python",
  $packer,
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
& $args[0] $args[1..($args.Count - 1)]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
