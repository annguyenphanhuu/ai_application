param(
    [string]$Categories = "All_Beauty",
    [int]$MaxProducts = 5000,
    [int]$MaxReviews = 20000,
    [string]$RawOutputDir = "data/raw/amazon_reviews_2023",
    [string]$ProcessedOutput = "data/processed/products_processed",
    [string]$TrackingUri = "sqlite:///mlflow.db",
    [string]$ExperimentName = "SmartShop_Rating_Classification",
    [string]$ModelName = "SmartShopRatingClassifier",
    [switch]$SkipDownload,
    [switch]$SkipEtl,
    [switch]$SkipTrain,
    [switch]$SkipCompose,
    [switch]$SkipIndex,
    [switch]$SkipSmoke,
    [switch]$UseSentenceTransformers
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

function Wait-HttpOk {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 120
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

if (-not $SkipSmoke) {
    Invoke-Step "Preflight tests and Docker config" {
        & "$PSScriptRoot\smoke_test.ps1" -SkipDockerBuild
    }
}

if (-not $SkipDownload) {
    Invoke-Step "Materialize Amazon Reviews 2023 sample" {
        python -m jobs.amazon_reviews_2023 `
            --categories $Categories `
            --max-products $MaxProducts `
            --max-reviews $MaxReviews `
            --output-dir $RawOutputDir
    }
}

if (-not $SkipEtl) {
    Invoke-Step "Run Spark ETL" {
        python -m jobs.spark_etl `
            --input-products "$RawOutputDir/combined/meta.jsonl" `
            --input-reviews "$RawOutputDir/combined/reviews.jsonl" `
            --output-path $ProcessedOutput `
            --output-format parquet `
            --master "local[*]"
    }
}

if (-not $SkipTrain) {
    Invoke-Step "Train and register MLflow candidate" {
        python -m src.train `
            --input-path $ProcessedOutput `
            --tracking-uri $TrackingUri `
            --experiment-name $ExperimentName `
            --model-name $ModelName `
            --register-model `
            --registry-alias candidate
    }

    Invoke-Step "Promote candidate to champion" {
        python -m src.model_registry promote `
            --tracking-uri $TrackingUri `
            --model-name $ModelName `
            --source-alias candidate `
            --alias champion
    }
}

if ($UseSentenceTransformers) {
    $env:SMARTSHOP_API_BUILD_TARGET = "full-runtime"
    $env:SMARTSHOP_EMBEDDING_BACKEND = "sentence-transformers"
    $env:SMARTSHOP_INDEX_EMBEDDING_BACKEND = "sentence-transformers"
}
else {
    $env:SMARTSHOP_API_BUILD_TARGET = "runtime"
    $env:SMARTSHOP_EMBEDDING_BACKEND = "hashing"
    $env:SMARTSHOP_INDEX_EMBEDDING_BACKEND = "hashing"
}

if (-not $SkipCompose) {
    Invoke-Step "Start Docker Compose runtime stack" {
        docker compose up --build -d
        docker compose ps
        Wait-HttpOk -Url "http://localhost:8000/health/ready" -TimeoutSeconds 180
    }
}

if (-not $SkipIndex) {
    Invoke-Step "Index products into Qdrant" {
        docker compose --profile indexer run --rm vector-indexer
        docker compose exec -T redis redis-cli FLUSHDB
    }
}

Invoke-Step "Probe runtime endpoints" {
    Wait-HttpOk -Url "http://localhost:8000/health" -TimeoutSeconds 60
    Wait-HttpOk -Url "http://localhost:8000/metrics" -TimeoutSeconds 60

    $token = docker compose exec -T api python -c "from src.main import create_access_token; print(create_access_token('admin-user'))"
    $headers = @{ Authorization = "Bearer $token" }
    Invoke-RestMethod -Headers $headers -Uri "http://localhost:8000/search?query=noise%20cancelling%20headphones&top_k=3"
}

Write-Host ""
Write-Host "End-to-end local production slice completed." -ForegroundColor Green

