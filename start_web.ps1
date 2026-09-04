param(
    [int]$Port = 8000,
    [switch]$SkipBuild,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$healthUrl = "http://127.0.0.1:$Port/api/system"

function Test-WebService {
    try {
        Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

if (-not $SkipBuild) {
    Push-Location "$projectRoot\web"
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    }
    finally {
        Pop-Location
    }
}

if (Test-WebService) {
    Write-Output "Voxel Studio is already running at http://127.0.0.1:$Port"
    exit 0
}

if ($Foreground) {
    Set-Location $projectRoot
    python -m uvicorn server:app --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
}

$logDir = "$projectRoot\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$pythonExe = (Get-Command python).Source
$serverArgs = @("-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "$Port")
$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $serverArgs `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$logDir\web.stdout.log" `
    -RedirectStandardError "$logDir\web.stderr.log" `
    -PassThru
$process.Id | Set-Content -LiteralPath "$projectRoot\.web-server.pid"

for ($attempt = 0; $attempt -lt 40; $attempt++) {
    Start-Sleep -Milliseconds 250
    if (Test-WebService) {
        Write-Output "Voxel Studio started at http://127.0.0.1:$Port (PID $($process.Id))"
        exit 0
    }
    if ($process.HasExited) { break }
}

$details = Get-Content -LiteralPath "$logDir\web.stderr.log" -Tail 20 -ErrorAction SilentlyContinue
throw "Voxel Studio failed to start. $details"

