param(
  [string]$OutDir = "",
  [ValidateSet("full","delta")]
  [string]$Mode = "full",
  [switch]$Zip = $true,
  [switch]$Combined = $true
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  $memoryDir = Split-Path -Parent $PSScriptRoot
  return (Resolve-Path (Split-Path -Parent $memoryDir)).Path
}

$repoRoot = Get-RepoRoot
$packer = Join-Path $repoRoot "MEMORY\\packer.py"

if (-not (Test-Path -LiteralPath $packer)) {
  throw "Missing Python packer at: $packer"
}

if ($OutDir -eq "") {
  $stamp = (Get-Date).ToString("yyyy-MM-dd_HH-mm-ss")
  $OutDir = "MEMORY/LLM-PACKER-1.0/_packs/llm-pack-$stamp"
}

$args = @(
  "python",
  $packer,
  "--mode", $Mode,
  "--out-dir", $OutDir
)

if ($Zip) { $args += "--zip" }
if ($Combined) { $args += "--combined" }

Write-Host "Running: $($args -join ' ')"
& $args[0] $args[1..($args.Count - 1)]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
