# Starts Grafana Alloy to push the API's /metrics to Grafana Cloud.
# Reads GRAFANA_CLOUD_URL, GRAFANA_CLOUD_USER, GRAFANA_CLOUD_API_KEY and ALLOY_PATH from .env.
# Usage (from the project root, with the API already running):  .\monitoring\run-alloy.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"

if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*(GRAFANA_CLOUD_URL|GRAFANA_CLOUD_USER|GRAFANA_CLOUD_API_KEY|ALLOY_PATH)\s*=\s*(.*)$') {
            $value = $Matches[2].Trim().Trim('"').Trim("'")
            if ($value) { Set-Item -Path "Env:$($Matches[1])" -Value $value }
        }
    }
}

$missing = @("GRAFANA_CLOUD_URL", "GRAFANA_CLOUD_USER", "GRAFANA_CLOUD_API_KEY") |
    Where-Object { -not (Get-Item "Env:$_" -ErrorAction SilentlyContinue).Value }
if ($missing) {
    Write-Host "Missing in .env: $($missing -join ', '). See docs/monitoring.md." -ForegroundColor Red
    exit 1
}

$candidates = @(
    $env:ALLOY_PATH,
    "D:\tools\alloy\alloy-windows-amd64.exe",
    "$env:ProgramFiles\GrafanaLabs\Alloy\alloy-windows-amd64.exe"  # winget / installer location
) | Where-Object { $_ }
$alloy = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $alloy) {
    Write-Host "Alloy not found. Install it (winget install GrafanaLabs.Alloy) or set ALLOY_PATH in .env." -ForegroundColor Red
    exit 1
}

try {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:8000/health" | Out-Null
} catch {
    Write-Host "Warning: the API is not responding on http://127.0.0.1:8000. Start it with: uv run uvicorn api:app" -ForegroundColor Yellow
}

# Keep Alloy's buffer inside the project (gitignored artifacts/) so it is writable without admin rights.
$storage = Join-Path $root "artifacts\alloy-data"
# Port 12346, because the Windows "Alloy" service installed by winget already uses 12345.
$listen = "127.0.0.1:12346"
Write-Host "Using $alloy" -ForegroundColor DarkGray
Write-Host "Alloy UI: http://$listen  (Ctrl+C to stop)" -ForegroundColor Green
& $alloy run (Join-Path $root "monitoring\alloy.alloy") "--storage.path=$storage" "--server.http.listen-addr=$listen"
