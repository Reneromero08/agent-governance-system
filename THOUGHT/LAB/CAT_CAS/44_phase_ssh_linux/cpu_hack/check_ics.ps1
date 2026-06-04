$mngr = New-Object -ComObject HNetCfg.HNetShare
foreach ($conn in $mngr.EnumEveryConnection) {
    $props = $mngr.NetConnectionProps($conn)
    $config = $mngr.INetSharingConfigurationForINetConnection($conn)
    $name = $props.Name
    $enabled = $config.SharingEnabled
    $type = $config.SharingConnectionType
    Write-Output "$name : Enabled=$enabled Type=$type"
}
