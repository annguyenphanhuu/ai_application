param(
    [switch]$SkipTests,
    [switch]$SkipDockerBuild,
    [switch]$StartStack,
    [switch]$SkipApiProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$ScriptBlock
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $ScriptBlock
}

function Test-CommandAvailable {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

function Wait-HttpOk {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                return
            }
        }
        catch {
            Start-Sleep -Seconds 3
        }
    } while ((Get-Date) -lt $deadline)

    throw "Timed out waiting for $Url"
}

Invoke-Step "Preflight commands" {
    Test-CommandAvailable python
    Test-CommandAvailable docker
}

if (-not $SkipTests) {
    Invoke-Step "Python tests" {
        python -m pytest -q
    }
}

Invoke-Step "Docker Compose config" {
    docker compose config | Out-Null
}

if (-not $SkipDockerBuild) {
    Invoke-Step "Docker runtime build" {
        docker build --target runtime -t smartshop-api:smoke .
    }
}

if ($StartStack) {
    Invoke-Step "Start runtime stack" {
        docker compose up --build -d api redis qdrant kafka prometheus grafana
        docker compose ps
    }

    if (-not $SkipApiProbe) {
        Invoke-Step "Probe API health and metrics" {
            Wait-HttpOk -Url "http://localhost:8000/health/ready" -TimeoutSeconds 120
            Wait-HttpOk -Url "http://localhost:8000/metrics" -TimeoutSeconds 30
            Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:8000/health" | Select-Object -ExpandProperty Content
        }
    }
}

Write-Host ""
Write-Host "Smoke test completed." -ForegroundColor Green

