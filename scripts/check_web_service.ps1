param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$url = "http://127.0.0.1:$Port/api/system"

try {
    $status = Invoke-RestMethod -Uri $url -TimeoutSec 3
}
catch {
    Write-Error "Web service is unavailable at $url"
    exit 1
}

if ($null -eq $status.checkpoints) {
    Write-Error "Web service returned an invalid status payload."
    exit 1
}

Write-Output "WEB_SERVICE_OK $($status.device) checkpoints=$($status.checkpoints.Count)"

