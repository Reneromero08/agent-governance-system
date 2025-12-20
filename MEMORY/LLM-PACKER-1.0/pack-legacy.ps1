param(
  [string]$OutDir = "",
  [string]$VaultPath = "",
  [switch]$Zip = $true,
  [switch]$Combined = $true
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
  # tools/LLM-PACKER-1.0 => tools => repo root
  $toolsDir = Split-Path -Parent $PSScriptRoot
  return (Resolve-Path (Split-Path -Parent $toolsDir)).Path
}

function Ensure-Dir([string]$path) {
  if (!(Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path | Out-Null
  }
}

function Is-TextExtension([string]$path) {
  $ext = ([IO.Path]::GetExtension($path)).ToLowerInvariant()
  if ($ext -eq "") {
    $leaf = Split-Path -Leaf $path
    return ($leaf -in @(".gitignore", ".gitattributes", ".editorconfig", ".htaccess", ".gitkeep"))
  }
  return $ext -in @(
    ".md", ".txt",
    ".json",
    ".py",
    ".js", ".mjs", ".cjs",
    ".css",
    ".html",
    ".php",
    ".ps1",
    ".cmd", ".bat",
    ".yml", ".yaml"
  )
}

function Copy-TextTree([string]$srcDir, [string]$dstDir, [string[]]$excludeRelGlobs, [ref]$omitted) {
  if (!(Test-Path -LiteralPath $srcDir)) { return }
  Ensure-Dir $dstDir

  $scopeName = Split-Path -Leaf $srcDir
  $items = Get-ChildItem -LiteralPath $srcDir -Recurse -Force -File
  foreach ($item in $items) {
    $rel = $item.FullName.Substring($srcDir.Length).TrimStart('\', '/')
    $skip = $false
    foreach ($g in $excludeRelGlobs) {
      if ($rel -like $g) { $skip = $true; break }
    }
    if ($skip) { continue }

    if (!(Is-TextExtension $item.FullName)) {
      $omitted.Value += [pscustomobject]@{
        scope = "repo"
        repoRelPath = ($scopeName + "/" + ($rel -replace '\\','/'))
        bytes = $item.Length
      }
      continue
    }

    $dest = Join-Path $dstDir $rel
    Ensure-Dir (Split-Path -Parent $dest)
    Copy-Item -Force -LiteralPath $item.FullName -Destination $dest
  }
}

function Read-Json([string]$path) {
  if (!(Test-Path -LiteralPath $path)) { return $null }
  return (Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json)
}

function Write-Json([object]$obj, [string]$path) {
  Ensure-Dir (Split-Path -Parent $path)
  ($obj | ConvertTo-Json -Depth 30) | Out-File -FilePath $path -Encoding utf8
}

function Get-Sha256([string]$path) {
  try { return (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash } catch { return $null }
}

function Parse-Frontmatter([string]$mdText) {
  # Returns @{ frontmatterText, bodyText, keys, values }
  $result = @{
    frontmatterText = ""
    bodyText = $mdText
    keys = @()
    values = @{}
  }

  if (-not $mdText.StartsWith("---")) { return $result }

  # Find the closing '---' on its own line
  $lines = $mdText -split "`r?`n"
  if ($lines.Count -lt 3) { return $result }
  if ($lines[0].Trim() -ne "---") { return $result }

  $endIndex = -1
  for ($i = 1; $i -lt $lines.Count; $i++) {
    if ($lines[$i].Trim() -eq "---") { $endIndex = $i; break }
  }
  if ($endIndex -lt 0) { return $result }

  $fmLines = $lines[1..($endIndex - 1)]
  $bodyLines = @()
  if ($endIndex + 1 -lt $lines.Count) { $bodyLines = $lines[($endIndex + 1)..($lines.Count - 1)] }

  $fmText = ($fmLines -join "`n")
  $result.frontmatterText = $fmText
  $result.bodyText = ($bodyLines -join "`n")

  foreach ($line in $fmLines) {
    if ($line -match '^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$') {
      $key = $matches[1]
      $val = $matches[2].Trim()
      $result.keys += $key
      if (-not $result.values.ContainsKey($key)) { $result.values[$key] = $val }
    }
  }

  return $result
}

function Extract-TokenLines([string]$bodyText) {
  $tokens = @()
  $lines = $bodyText -split "`r?`n"
  for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    if ($line -match '^\s*\[[^\]]+\]\s+#\w+\b') {
      $tokens += [pscustomobject]@{ line = ($i + 1); kind = "bracket-directive"; text = $line.TrimEnd() }
      continue
    }
    if ($line -match '^\s*#(altmap|capmap|favorites)\b') {
      $tokens += [pscustomobject]@{ line = ($i + 1); kind = "map-or-directive"; text = $line.TrimEnd() }
      continue
    }
  }
  return $tokens
}

function Build-StartHere([string]$path) {
  $content = @()
  $content += "# START HERE"
  $content += ""
  $content += "This snapshot is meant to be shared with any LLM to continue work on the Agent Governance System (AGS) repository."
  $content += ""
  $content += "## Read order"
  $content += '1) `repo/AGENTS.md` (procedural operating contract)'
  $content += '2) `repo/README.md` + `repo/ROADMAP.md` (orientation)'
  $content += '3) `repo/CANON/CONTRACT.md` + `repo/CANON/INVARIANTS.md` + `repo/CANON/VERSIONING.md` (authority)'
  $content += '4) `repo/MAPS/ENTRYPOINTS.md` (where to change what)'
  $content += '5) `repo/CONTRACTS/runner.py` + `repo/SKILLS/` (execution + fixtures)'
  $content += '6) `meta/ENTRYPOINTS.md` (snapshot-specific pointers)'
  $content += ""
  $content += "## Notes"
  $content += "- `BUILD/` is not included (generated output root)."
  $content += "- Research under `repo/CONTEXT/research/` is non-binding and opt-in."
  $content += '- Any optional external markdown included via -VaultPath is copied under `vault/`.'
  $content += '- Binary files inside the repo are omitted; see `meta/REPO_OMITTED_BINARIES.json`.'
  $content += ""
  $content -join "`n" | Out-File -FilePath $path -Encoding utf8
}

function Build-Entrypoints([string]$path) {
  $content = @()
  $content += "# Entrypoints (Where To Change What)"
  $content += ""
  $content += "## Orientation"
  $content += '- `repo/README.md`'
  $content += '- `repo/ROADMAP.md`'
  $content += ""
  $content += "## Operating contract"
  $content += '- `repo/AGENTS.md`'
  $content += ""
  $content += "## Canon (authority)"
  $content += '- `repo/CANON/CONTRACT.md`'
  $content += '- `repo/CANON/INVARIANTS.md`'
  $content += '- `repo/CANON/VERSIONING.md`'
  $content += '- `repo/CANON/GLOSSARY.md`'
  $content += '- `repo/CANON/SECURITY.md`'
  $content += '- `repo/CANON/CHANGELOG.md`'
  $content += '- `repo/CANON/AGENTS.md`'
  $content += ""
  $content += "## Maps (navigation)"
  $content += '- `repo/MAPS/ENTRYPOINTS.md`'
  $content += '- `repo/MAPS/SYSTEM_MAP.md`'
  $content += '- `repo/MAPS/DATA_FLOW.md`'
  $content += '- `repo/MAPS/FILE_OWNERSHIP.md`'
  $content += ""
  $content += "## Context (non-canon records)"
  $content += '- `repo/CONTEXT/INDEX.md`'
  $content += '- `repo/CONTEXT/decisions/ADR-000-template.md`'
  $content += '- `repo/CONTEXT/open/OPEN-000-template.md`'
  $content += '- `repo/CONTEXT/preferences/STYLE-000-template.md`'
  $content += '- `repo/CONTEXT/rejected/REJECT-000-template.md`'
  $content += '- `repo/CONTEXT/research/INDEX.md`'
  $content += ""
  $content += "## Skills and contracts"
  $content += '- `repo/SKILLS/`'
  $content += '- `repo/CONTRACTS/runner.py`'
  $content += '- `repo/CONTRACTS/fixtures/`'
  $content += '- `repo/CONTRACTS/schemas/`'
  $content += ""
  $content += "## Cortex and tools"
  $content += '- `repo/CORTEX/cortex.build.py`'
  $content += '- `repo/CORTEX/query.py`'
  $content += '- `repo/TOOLS/`'
  $content += ""
  $content += "## Memory"
  $content += '- `repo/MEMORY/packer.py`'
  $content += ""
  $content -join "`n" | Out-File -FilePath $path -Encoding utf8
}

function New-TreeNode {
  return @{
    dirs = @{}
    files = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::OrdinalIgnoreCase)
  }
}

function Add-TreePath([hashtable]$root, [string]$relPath) {
  if (-not $relPath) { return }
  $parts = $relPath.Split("/", [System.StringSplitOptions]::RemoveEmptyEntries)
  if ($parts.Count -eq 0) { return }

  $node = $root
  for ($i = 0; $i -lt $parts.Count; $i++) {
    $part = $parts[$i]
    $isLeaf = ($i -eq $parts.Count - 1)
    if ($isLeaf) {
      [void]$node.files.Add($part)
      return
    }
    if (-not $node.dirs.ContainsKey($part)) {
      $node.dirs[$part] = (New-TreeNode)
    }
    $node = $node.dirs[$part]
  }
}

function Render-Tree([hashtable]$node, [string]$prefix, [System.Collections.Generic.List[string]]$outLines) {
  $dirNames = @($node.dirs.Keys | Sort-Object)
  $fileNames = @($node.files | Sort-Object)

  $entries = @()
  foreach ($d in $dirNames) { $entries += [pscustomobject]@{ type = "dir"; name = $d } }
  foreach ($f in $fileNames) { $entries += [pscustomobject]@{ type = "file"; name = $f } }

  for ($i = 0; $i -lt $entries.Count; $i++) {
    $e = $entries[$i]
    $isLast = ($i -eq $entries.Count - 1)
    $connector = $(if ($isLast) { '\-- ' } else { '|-- ' })
    $childPrefix = $prefix + $(if ($isLast) { '    ' } else { '|   ' })

    if ($e.type -eq "dir") {
      $outLines.Add($prefix + $connector + $e.name + "/") | Out-Null
      Render-Tree -node $node.dirs[$e.name] -prefix $childPrefix -outLines $outLines
    } else {
      $outLines.Add($prefix + $connector + $e.name) | Out-Null
    }
  }
}

function Build-PackTreeText([string[]]$paths, [string[]]$extraPaths) {
  $treeRoot = New-TreeNode
  foreach ($p in $paths) { Add-TreePath -root $treeRoot -relPath $p }
  foreach ($p in $extraPaths) { Add-TreePath -root $treeRoot -relPath $p }

  $treeLines = New-Object System.Collections.Generic.List[string]
  $treeLines.Add("PACK/") | Out-Null
  Render-Tree -node $treeRoot -prefix "" -outLines $treeLines
  return ($treeLines -join "`n")
}

function Get-Lang([string]$path) {
  $ext = ([IO.Path]::GetExtension($path)).ToLowerInvariant()
  switch ($ext) {
    ".py" { return "python" }
    ".js" { return "js" }
    ".mjs" { return "js" }
    ".cjs" { return "js" }
    ".css" { return "css" }
    ".html" { return "html" }
    ".md" { return "md" }
    ".json" { return "json" }
    ".yml" { return "yaml" }
    ".yaml" { return "yaml" }
    ".ps1" { return "powershell" }
    ".cmd" { return "bat" }
    ".bat" { return "bat" }
    ".php" { return "php" }
    default { return "" }
  }
}

function Get-Fence([string]$text) {
  $max = 0
  foreach ($m in [regex]::Matches($text, '`+')) {
    if ($m.Value.Length -gt $max) { $max = $m.Value.Length }
  }
  $len = [Math]::Max(3, $max + 1)
  return ('`' * $len)
}

function Append-SourceToMarkdown([System.Text.StringBuilder]$sb, [string]$packRelPath, [string]$absPath) {
  $fi = Get-Item -LiteralPath $absPath
  $text = Get-Content -LiteralPath $absPath -Raw -Encoding utf8
  $lang = Get-Lang $packRelPath
  $fence = Get-Fence $text
  $fenceOpen = $(if ($lang -ne "") { $fence + $lang } else { $fence })

  [void]$sb.AppendLine("")
  [void]$sb.AppendLine("-----")
  [void]$sb.AppendLine("Source: " + ('`' + $packRelPath + '`'))
  [void]$sb.AppendLine("Bytes: " + $fi.Length)
  [void]$sb.AppendLine("-----")
  [void]$sb.AppendLine("")
  [void]$sb.AppendLine($fenceOpen)
  [void]$sb.AppendLine($text.TrimEnd())
  [void]$sb.AppendLine($fence)
}

function Write-SplitPack(
  [string]$packDir,
  [string[]]$allPackPathsInOrder,
  [string]$fullCombinedRelPath
) {
  $splitDir = Join-Path $packDir "COMBINED\\SPLIT"
  Ensure-Dir $splitDir

  function Abs([string]$rel) { return (Join-Path $packDir ($rel -replace '/', '\')) }
  function Exists([string]$rel) { return (Test-Path -LiteralPath (Abs $rel)) }
  function MdCode([string]$s) { return ('`' + $s + '`') }

  function PathsByPrefix([string]$prefix) {
    return @($allPackPathsInOrder | Where-Object { $_.StartsWith($prefix) })
  }

  function Require([string[]]$rels) {
    foreach ($r in $rels) {
      if (-not (Exists $r)) { throw "Split pack missing required source: $r" }
    }
  }

  # Define sources (keep ordering stable, using the same pack ordering where possible)
  $canonSources = @(
    "repo/AGENTS.md",
    "repo/CANON/CONTRACT.md",
    "repo/CANON/INVARIANTS.md",
    "repo/CANON/VERSIONING.md",
    "repo/CANON/GLOSSARY.md",
    "repo/CANON/SECURITY.md",
    "repo/CANON/CHANGELOG.md",
    "repo/CANON/AGENTS.md"
  )

  $orientationSources = @(
    "repo/README.md",
    "repo/ROADMAP.md"
  )

  $mapsSources = @(
    "meta/START_HERE.md",
    "meta/ENTRYPOINTS.md",
    "repo/MAPS/ENTRYPOINTS.md",
    "repo/MAPS/SYSTEM_MAP.md",
    "repo/MAPS/DATA_FLOW.md",
    "repo/MAPS/FILE_OWNERSHIP.md"
  )

  $contextSources = @(
    "repo/CONTEXT/INDEX.md",
    "repo/CONTEXT/decisions/ADR-000-template.md",
    "repo/CONTEXT/open/OPEN-000-template.md",
    "repo/CONTEXT/preferences/STYLE-000-template.md",
    "repo/CONTEXT/rejected/REJECT-000-template.md",
    "repo/CONTEXT/research/INDEX.md"
  )

  $skillsSources = @(
    "repo/SKILLS/_TEMPLATE/SKILL.md",
    "repo/SKILLS/_TEMPLATE/validate.py",
    "repo/SKILLS/_TEMPLATE/run.sh",
    "repo/SKILLS/example-echo/SKILL.md",
    "repo/SKILLS/example-echo/run.py",
    "repo/SKILLS/example-echo/validate.py"
  )
  $skillsSources += (PathsByPrefix "repo/SKILLS/example-echo/fixtures/")

  $contractsSources = @(
    "repo/CONTRACTS/README.md",
    "repo/CONTRACTS/runner.py"
  )
  $contractsSources += (PathsByPrefix "repo/CONTRACTS/fixtures/")
  $contractsSources += (PathsByPrefix "repo/CONTRACTS/schemas/")

  $cortexToolsMemorySources = @()
  $cortexToolsMemorySources += (PathsByPrefix "repo/CORTEX/")
  $cortexToolsMemorySources += (PathsByPrefix "repo/TOOLS/")
  $cortexToolsMemorySources += (PathsByPrefix "repo/MEMORY/")

  $appendixSources = @(
    "meta/FILE_TREE.txt",
    "meta/FILE_INDEX.json",
    "meta/REPO_OMITTED_BINARIES.json",
    "meta/VAULT_PAGES_INDEX.json",
    "meta/VAULT_TOKENS_INDEX.json",
    "meta/VAULT_ASSETS_INVENTORY.json",
    "meta/VAULT_NOT_INCLUDED.txt"
  )

  # Validate critical sources exist.
  Require $canonSources
  Require $orientationSources
  Require $mapsSources
  Require @("repo/CONTRACTS/runner.py")
  Require @("repo/SKILLS/_TEMPLATE/SKILL.md")

  $chunks = @(
    @{
      file = "01_CANON.md"
      title = "Canon and Agent Contract"
      about = "Authority sources: canon plus the operating contract."
      sources = $canonSources
      priority = 1
    },
    @{
      file = "02_ORIENTATION.md"
      title = "Orientation"
      about = "Project overview and roadmap."
      sources = $orientationSources
      priority = 2
    },
    @{
      file = "03_MAPS_ENTRYPOINTS.md"
      title = "Maps and Entrypoints"
      about = "Navigation maps and entrypoints for making changes."
      sources = $mapsSources
      priority = 3
    },
    @{
      file = "04_CONTEXT.md"
      title = "Context Records"
      about = "Context index and templates (non-binding guidance)."
      sources = $contextSources
      priority = 4
    },
    @{
      file = "05_SKILLS_CONTRACTS.md"
      title = "Skills and Contracts"
      about = "Skill template, reference skill, and contract runner plus schemas/fixtures."
      sources = ($skillsSources + $contractsSources)
      priority = 5
    },
    @{
      file = "06_CORTEX_TOOLS_MEMORY.md"
      title = "Cortex, Tools, and Memory"
      about = "Cortex index/query, governance tools, and memory packer."
      sources = $cortexToolsMemorySources
      priority = 6
    },
    @{
      file = "07_APPENDIX.md"
      title = "Appendix"
      about = "Snapshot indices and optional vault inventories."
      sources = $appendixSources
      priority = 7
    }
  )

  if ($chunks.Count -gt 7) { throw "Split pack must be <= 7 chunks (+ index). Got $($chunks.Count)." }

  # Write chunks + manifest
  $manifest = @()
  foreach ($chunk in $chunks) {
    $outPath = Join-Path $splitDir $chunk.file
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine("# " + $chunk.title)
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine($chunk.about)

    foreach ($srcRel in $chunk.sources) {
      if (-not (Exists $srcRel)) {
        # For appendix indices, allow missing (e.g. vault not included) by skipping quietly.
        continue
      }
      Append-SourceToMarkdown -sb $sb -packRelPath $srcRel -absPath (Abs $srcRel)
    }

    $sb.ToString() | Out-File -FilePath $outPath -Encoding utf8
    $manifest += [pscustomobject]@{ file = $chunk.file; title = $chunk.title; priority = $chunk.priority; sources = $chunk.sources }
  }

  # Write index (must match outputs)
  $indexPath = Join-Path $splitDir "00_INDEX.md"
  $indexLines = @()
  $indexLines += "# Combined Split Pack"
  $indexLines += ""
  $indexLines += "Purpose: This is a split, LLM-friendly version of the combined snapshot so a model can load the system in priority order without burning tokens. It is generated from the same sources and preserves verbatim content; the full combined remains available for exhaustive search."
  $indexLines += ""
  $indexLines += ("Full combined (unchanged): " + (MdCode $fullCombinedRelPath))
  $indexLines += ""
  $indexLines += "## Load Order"
  $indexLines += ""
  $indexLines += "| Priority | File | What's Inside | Authoritative Sources |"
  $indexLines += "|---:|---|---|---|"
  foreach ($chunk in ($chunks | Sort-Object { [int]$_.priority })) {
    $srcList = ($chunk.sources | ForEach-Object { MdCode $_ }) -join '<br>'
    $indexLines += ("| " + $chunk.priority + " | " + (MdCode ("COMBINED/SPLIT/" + $chunk.file)) + " | " + $chunk.about + " | " + $srcList + " |")
  }
  $indexLines += ""
  $indexLines += "## Manifest"
  $indexLines += ""
  $indexLines += '```json'
  $indexLines += ($manifest | ConvertTo-Json -Depth 10)
  $indexLines += '```'

  ($indexLines -join "`n") | Out-File -FilePath $indexPath -Encoding utf8

  # Validate <= 8 files and index references exist
  $actual = Get-ChildItem -LiteralPath $splitDir -File | Select-Object -ExpandProperty Name
  if ($actual.Count -gt 8) { throw "Split pack must be <= 8 files. Got $($actual.Count): $($actual -join ', ')" }
  if (-not ($actual -contains "00_INDEX.md")) { throw "Split pack missing 00_INDEX.md" }
  foreach ($chunk in $chunks) {
    if (-not ($actual -contains $chunk.file)) { throw "Split pack missing chunk file: $($chunk.file)" }
  }
}

function Write-CombinedOutputs(
  [string]$packDir,
  [string[]]$orderedPaths,
  [string]$treePath
) {
  $combinedDir = Join-Path $packDir "COMBINED"
  Ensure-Dir $combinedDir

  $mdOut = Join-Path $combinedDir "AGS_COMBINED.md"
  $txtOut = Join-Path $combinedDir "AGS_COMBINED.txt"
  $treeMdOut = Join-Path $combinedDir "TREE.md"
  $treeTxtOut = Join-Path $combinedDir "TREE.txt"

  $treeText = Build-PackTreeText -paths $orderedPaths -extraPaths @(
    "COMBINED/TREE.md",
    "COMBINED/TREE.txt",
    "COMBINED/AGS_COMBINED.md",
    "COMBINED/AGS_COMBINED.txt"
  )
  $treeText | Out-File -FilePath $treeTxtOut -Encoding utf8

  $treeMdLines = @()
  $treeMdLines += "# PACK TREE"
  $treeMdLines += ""
  $treeMdLines += '```text'
  $treeMdLines += $treeText.TrimEnd()
  $treeMdLines += '```'
  ($treeMdLines -join "`n") | Out-File -FilePath $treeMdOut -Encoding utf8

  $header = @()
  $header += "# AGS_COMBINED"
  $header += ""
  $header += "Generated: " + (Get-Date).ToString("s")
  $header += "Includes: repo (text only) + optional vault markdown + meta indices"
  $header += ""
  $headerText = ($header -join "`n")

  $mdSb = New-Object System.Text.StringBuilder
  $txtSb = New-Object System.Text.StringBuilder
  [void]$mdSb.AppendLine($headerText)
  [void]$txtSb.AppendLine($headerText)

  $includedCount = 0
  $includedBytes = 0

  foreach ($rel in $orderedPaths) {
    if ($rel -like "COMBINED/*") { continue }
    if ($rel -like "*.zip") { continue }

    $full = Join-Path $packDir ($rel -replace '/', '\')
    if (!(Test-Path -LiteralPath $full)) { continue }

    $fi = Get-Item -LiteralPath $full
    # Only include textual files in the combined output.
    if (!(Is-TextExtension $full)) {
      continue
    }

    $text = Get-Content -LiteralPath $full -Raw -Encoding utf8

    # TXT: simple separators
    [void]$txtSb.AppendLine("")
    [void]$txtSb.AppendLine("-----")
    [void]$txtSb.AppendLine("FILE: " + $rel)
    [void]$txtSb.AppendLine("BYTES: " + $fi.Length)
    [void]$txtSb.AppendLine("-----")
    [void]$txtSb.AppendLine("")
    [void]$txtSb.AppendLine($text.TrimEnd())

    # MD: file header + fenced code block for safety/readability
    $lang = Get-Lang $rel
    $fence = Get-Fence $text
    $fenceOpen = $(if ($lang -ne "") { $fence + $lang } else { $fence })
    [void]$mdSb.AppendLine("")
    [void]$mdSb.AppendLine("## " + ('`' + $rel + '`'))
    [void]$mdSb.AppendLine("")
    [void]$mdSb.AppendLine("- bytes: " + $fi.Length)
    [void]$mdSb.AppendLine("")
    [void]$mdSb.AppendLine($fenceOpen)
    [void]$mdSb.AppendLine($text.TrimEnd())
    [void]$mdSb.AppendLine($fence)

    $includedCount++
    $includedBytes += $fi.Length
  }

  $mdSb.ToString() | Out-File -FilePath $mdOut -Encoding utf8
  $txtSb.ToString() | Out-File -FilePath $txtOut -Encoding utf8
}

$repoRoot = Get-RepoRoot
$packerRoot = $PSScriptRoot

$buildRoot = Join-Path $repoRoot "BUILD"
$packsRoot = Join-Path $buildRoot "llm-packs"
Ensure-Dir $packsRoot

if ($OutDir -eq "") {
  $stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
  $OutDir = Join-Path $packsRoot ("llm-pack-" + $stamp)
} else {
  if (-not [IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $repoRoot $OutDir
  }
  $OutDir = [IO.Path]::GetFullPath($OutDir)
}

$outDirFull = [IO.Path]::GetFullPath($OutDir)
$buildFull = [IO.Path]::GetFullPath($buildRoot).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $outDirFull.StartsWith($buildFull, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "OutDir must be under BUILD/. Received: $outDirFull"
}
$OutDir = $outDirFull

Ensure-Dir $OutDir
Ensure-Dir (Join-Path $OutDir "repo")
Ensure-Dir (Join-Path $OutDir "vault")
Ensure-Dir (Join-Path $OutDir "meta")

$omittedRepo = @()

# ---- Copy repo (text only)

$repoOut = Join-Path $OutDir "repo"

$copyDirs = @("CANON", "CONTEXT", "MAPS", "SKILLS", "CONTRACTS", "MEMORY", "CORTEX", "TOOLS", ".github")
foreach ($d in $copyDirs) {
  $src = Join-Path $repoRoot $d
  $dst = Join-Path $repoOut $d
  if (!(Test-Path -LiteralPath $src)) { continue }

  $exclude = @()
  if ($d -eq "TOOLS") {
    # Do not pack previously generated packs from older runs.
    $exclude = @("LLM-PACKER-1.0\_packs\*")
  }
  Copy-TextTree $src $dst $exclude ([ref]$omittedRepo)
}

$rootFiles = @(
  "README.md",
  "ROADMAP.md",
  "LICENSE",
  "AGENTS.md",
  ".gitignore",
  ".gitattributes",
  ".editorconfig"
)
foreach ($f in $rootFiles) {
  $src = Join-Path $repoRoot $f
  if (!(Test-Path -LiteralPath $src)) { continue }
  if (-not (Is-TextExtension $src)) { continue }
  Copy-Item -Force -LiteralPath $src -Destination (Join-Path $repoOut $f)
}

Write-Json $omittedRepo (Join-Path $OutDir "meta\\REPO_OMITTED_BINARIES.json")

# ---- Optional: external vault (markdown only) + indices

$vaultRoot = $null
if ($VaultPath -ne "") {
  try { $vaultRoot = (Resolve-Path $VaultPath).Path } catch { $vaultRoot = $null }
}

$vaultPagesIndex = @()
$vaultTokensIndex = @()
$vaultAssetsInventory = @()

if ($null -ne $vaultRoot -and (Test-Path -LiteralPath $vaultRoot)) {
  $vaultOut = Join-Path $OutDir "vault"

  $allVaultFiles = Get-ChildItem -LiteralPath $vaultRoot -Recurse -Force -File
  foreach ($file in $allVaultFiles) {
    $rel = $file.FullName.Substring($vaultRoot.Length).TrimStart('\', '/') -replace '\\','/'

    if ($file.Extension.ToLowerInvariant() -eq ".md") {
      $mdText = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8
      $fm = Parse-Frontmatter $mdText
      $tokens = Extract-TokenLines $fm.bodyText

      $vaultPagesIndex += [pscustomobject]@{
        relPath = $rel
        title = $(if ($fm.values.ContainsKey("title")) { $fm.values["title"] } else { "" })
        frontmatterKeys = ($fm.keys | Select-Object -Unique)
      }

      $vaultTokensIndex += [pscustomobject]@{
        relPath = $rel
        tokens = $tokens
      }

      $dst = Join-Path $vaultOut ($rel -replace '/', '\')
      Ensure-Dir (Split-Path -Parent $dst)
      Copy-Item -Force -LiteralPath $file.FullName -Destination $dst
      continue
    }

    $vaultAssetsInventory += [pscustomobject]@{
      relPath = $rel
      bytes = $file.Length
    }
  }

  Write-Json $vaultPagesIndex (Join-Path $OutDir "meta\\VAULT_PAGES_INDEX.json")
  Write-Json $vaultTokensIndex (Join-Path $OutDir "meta\\VAULT_TOKENS_INDEX.json")
  Write-Json $vaultAssetsInventory (Join-Path $OutDir "meta\\VAULT_ASSETS_INVENTORY.json")
} else {
  $note = @(
    "Vault not included.",
    "Provide -VaultPath to include external markdown under vault/."
  ) -join "`n"
  $note | Out-File -FilePath (Join-Path $OutDir "meta\\VAULT_NOT_INCLUDED.txt") -Encoding utf8
}

# ---- Meta: start here + entrypoints

Build-StartHere (Join-Path $OutDir "meta\\START_HERE.md")
Build-Entrypoints (Join-Path $OutDir "meta\\ENTRYPOINTS.md")

# ---- Meta: file tree + file index (pack contents)

$allFiles = Get-ChildItem -LiteralPath $OutDir -Recurse -Force -File
$tree = $allFiles | ForEach-Object { $_.FullName.Substring($OutDir.Length).TrimStart('\', '/') -replace '\\','/' } | Sort-Object
$treePath = Join-Path $OutDir "meta\\FILE_TREE.txt"
$treeText = Build-PackTreeText -paths $tree -extraPaths @()
$treeText | Out-File -FilePath $treePath -Encoding utf8

$fileIndex = @()
foreach ($file in $allFiles) {
  $rel = $file.FullName.Substring($OutDir.Length).TrimStart('\', '/') -replace '\\','/'
  $fileIndex += [pscustomobject]@{
    path = $rel
    bytes = $file.Length
    modified = $file.LastWriteTimeUtc.ToString("s") + "Z"
    sha256 = $(if ($file.Length -le 2MB) { Get-Sha256 $file.FullName } else { $null })
  }
}
$fileIndexPath = Join-Path $OutDir "meta\\FILE_INDEX.json"
Write-Json $fileIndex $fileIndexPath

# ---- Optional: COMBINED outputs (single MD + TXT + tree)

if ($Combined) {
  $orderedPaths = $tree
  Write-CombinedOutputs -packDir $OutDir -orderedPaths $orderedPaths -treePath $treePath
}

# ---- Split combined pack (LLM-friendly, <= 8 files)

Write-SplitPack -packDir $OutDir -allPackPathsInOrder $tree -fullCombinedRelPath "COMBINED/AGS_COMBINED.md"

# ---- Manifest

$manifest = [pscustomobject]@{
  packerVersion = "1.0"
  generatedAt = (Get-Date).ToString("s")
  repoRoot = "<REPO_ROOT>"
  vaultRoot = "<VAULT_ROOT>"
  includes = @{
    repoDirs = $copyDirs
    repoRootFiles = $rootFiles
    vaultMd = ($null -ne $vaultRoot)
    vaultAssets = "inventory only"
    combined = [bool]$Combined
  }
  excludes = @(
    "BUILD/** (generated output root)",
    "node_modules/**",
    "vault non-markdown files (inventory only)"
  )
}
Write-Json $manifest (Join-Path $OutDir "MANIFEST.json")

# ---- Zip

if ($Zip) {
  $archiveDir = Join-Path $packsRoot "archive"
  Ensure-Dir $archiveDir

  # Keep BUILD/llm-packs tidy: move any existing zips in the root into archive.
  Get-ChildItem -LiteralPath $packsRoot -File -Filter "*.zip" -ErrorAction SilentlyContinue | ForEach-Object {
    $dest = Join-Path $archiveDir $_.Name
    if (Test-Path -LiteralPath $dest) { Remove-Item -Force -LiteralPath $dest }
    Move-Item -Force -LiteralPath $_.FullName -Destination $dest
  }

  $zipPath = $OutDir.TrimEnd('\') + ".zip"
  if (Test-Path -LiteralPath $zipPath) { Remove-Item -Force -LiteralPath $zipPath }
  Compress-Archive -Path (Join-Path $OutDir "*") -DestinationPath $zipPath

  $archivedZip = Join-Path $archiveDir (Split-Path -Leaf $zipPath)
  if (Test-Path -LiteralPath $archivedZip) { Remove-Item -Force -LiteralPath $archivedZip }
  Move-Item -Force -LiteralPath $zipPath -Destination $archivedZip
  Write-Host "Created zip: $archivedZip"
}

Write-Host "Pack created: $OutDir"
