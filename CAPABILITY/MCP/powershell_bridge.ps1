<#
.SYNOPSIS
    Local PowerShell bridge for controlled command execution.

.DESCRIPTION
    Starts a local HTTP listener that accepts POST /run requests with JSON payloads
    and executes PowerShell commands. Intended for localhost use only.

SECURITY
    - Requires a shared token via X-Bridge-Token header (or ?token=...)
    - Optional allowlist via allowed_prefixes
    - Binds to 127.0.0.1 by default
#>

param(
    [string]$ConfigPath = "",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Read-Config {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        $Path = Join-Path $PSScriptRoot "powershell_bridge_config.json"
    }
    if (-not (Test-Path $Path)) {
        throw "CONFIG_NOT_FOUND: $Path"
    }
    return Get-Content $Path | ConvertFrom-Json
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $logDir = Join-Path $repoRoot "LAW\CONTRACTS\_runs\mcp_logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logPath = Join-Path $logDir "powershell_bridge.log"
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$ts] [$Level] $Message"
}

function Send-Response {
    param([System.Net.HttpListenerResponse]$Response, [hashtable]$Payload)
    $json = ($Payload | ConvertTo-Json -Depth 6)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $Response.ContentType = "application/json"
    $Response.ContentLength64 = $bytes.Length
    $Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $Response.OutputStream.Close()
}

$config = Read-Config $ConfigPath
$listenHost = "127.0.0.1"
if ($config.listen_host) { $listenHost = [string]$config.listen_host }
$port = [int]$config.port
$token = [string]$config.token
$allowed = @()
if ($config.allowed_prefixes) {
    $allowed = @($config.allowed_prefixes)
}

if ($token -eq "CHANGE_ME") {
    Write-Log "Token is still CHANGE_ME. Update powershell_bridge_config.json" "WARN"
}

$listener = New-Object System.Net.HttpListener
$prefixHost = $listenHost
if ($listenHost -eq "0.0.0.0" -or $listenHost -eq "*" -or $listenHost -eq "+") {
    $prefixHost = "+"
}
$prefix = "http://$prefixHost`:$port/"
$listener.Prefixes.Add($prefix)
try {
    $listener.Start()
} catch {
    Write-Log "HttpListener start failed for prefix $prefix. Try running as admin and reserving URL ACL." "ERROR"
    Write-Log "Example (admin): netsh http add urlacl url=$prefix user=$env:USERNAME" "ERROR"
    throw
}
Write-Log "PowerShell bridge listening on $prefix" "INFO"

try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        $response = $context.Response

        try {
            if ($request.HttpMethod -ne "POST" -or $request.Url.AbsolutePath -ne "/run") {
                throw "NOT_FOUND"
            }

            $tokenHeader = $request.Headers["X-Bridge-Token"]
            $tokenQuery = $request.QueryString["token"]
            if ($token -and $token -ne "CHANGE_ME") {
                if ($tokenHeader -ne $token -and $tokenQuery -ne $token) {
                    throw "UNAUTHORIZED"
                }
            }

            $reader = New-Object System.IO.StreamReader($request.InputStream, $request.ContentEncoding)
            $body = $reader.ReadToEnd()
            if ([string]::IsNullOrWhiteSpace($body)) {
                throw "EMPTY_BODY"
            }

            $payload = $body | ConvertFrom-Json
            $command = [string]$payload.command
            if ([string]::IsNullOrWhiteSpace($command)) {
                throw "COMMAND_REQUIRED"
            }

            if ($allowed.Count -gt 0) {
                $matched = $false
                foreach ($prefix in $allowed) {
                    if ($command.StartsWith($prefix)) { $matched = $true; break }
                }
                if (-not $matched) { throw "COMMAND_NOT_ALLOWED" }
            }

            $output = ""
            $hadError = $false
            $exitCode = 0

            if ($payload.cwd) { Push-Location -Path $payload.cwd }
            try {
                $output = Invoke-Expression $command 2>&1 | Out-String
                if ($LASTEXITCODE -ne $null) { $exitCode = [int]$LASTEXITCODE }
            } catch {
                $hadError = $true
                $output = $_ | Out-String
                $exitCode = 1
            } finally {
                if ($payload.cwd) { Pop-Location }
            }

            $result = @{ ok = (-not $hadError); exit_code = $exitCode; output = $output.TrimEnd() }
            Send-Response $response $result
            Write-Log "OK: $command" "INFO"

            if ($Once) { break }
        } catch {
            $err = $_.Exception.Message
            Send-Response $response @{ ok = $false; error = $err }
            Write-Log "ERROR: $err" "ERROR"
        }
    }
} finally {
    $listener.Stop()
    $listener.Close()
}
