$service = "GTAlgo"

try {
    $svc = Get-Service -Name $service -ErrorAction Stop
    if ($svc.Status -ne "Running") {
        Write-Host "Service $service is not running (Current status: $($svc.Status)). Waiting 15s before starting..."
        Start-Sleep -Seconds 15
        Start-Service -Name $service
        Write-Host "Service $service start command completed."
    }
}
catch {
    Write-Warning "Watchdog failed to check/start service ${service}: $_"
}