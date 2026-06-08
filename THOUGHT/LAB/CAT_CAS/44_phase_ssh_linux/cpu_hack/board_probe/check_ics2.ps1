$ErrorActionPreference = "Stop"
try {
    $mngr = New-Object -ComObject HNetCfg.HNetShare
    if ($mngr.EnumEveryConnection -eq $null) {
        Write-Output "EnumEveryConnection is null"
    } else {
        foreach ($conn in $mngr.EnumEveryConnection) {
            $props = $mngr.NetConnectionProps($conn)
            $config = $mngr.INetSharingConfigurationForINetConnection($conn)
            $name = $props.Name
            $enabled = $config.SharingEnabled
            $type = $config.SharingConnectionType
            Write-Output "$name : Enabled=$enabled Type=$type"
        }
    }
} catch {
    Write-Output "Error: $_"
}
