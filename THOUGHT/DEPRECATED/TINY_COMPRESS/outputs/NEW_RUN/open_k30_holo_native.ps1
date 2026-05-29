$ErrorActionPreference = "Stop"

$RepoRoot = "D:\CCC 2.0\AI\agent-governance-system"
$Pythonw = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
$Viewer = Join-Path $RepoRoot "THOUGHT\LAB\TINY_COMPRESS\holographic-image\holo_open.pyw"
$Holo = Join-Path $PSScriptRoot "FULLRES_rendered_catcas_topology_k65536.holo"

Start-Process -FilePath $Pythonw -ArgumentList @($Viewer, $Holo) -WorkingDirectory (Split-Path $Viewer)
