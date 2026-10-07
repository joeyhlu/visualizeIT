param()
$ErrorActionPreference = 'Stop'
$stableRoot = Split-Path -Parent $PSScriptRoot
$stablePython = Join-Path $stableRoot '.cache/quality-windows/Scripts/python.exe'
Push-Location -LiteralPath $stableRoot
try {
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    $env:NUMEXPR_NUM_THREADS='1'; $env:PYTHONHASHSEED='0'
    Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue
    $activeModels=Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_runner'
    }
    if ($activeModels) { throw 'Another neural experiment is active.' }
    $stableOutput='.cache/model-quality/results/keyboard/render-stability'
    $shortReport=Get-Content -LiteralPath "$stableOutput/report.json" -Raw | ConvertFrom-Json
    if (-not $shortReport.prefix_identical) { throw 'Short real prefix gate must pass first.' }
    if (Test-Path "$stableOutput/prefix-30.json") { throw 'Preserve prior output; choose a new experiment directory.' }
    # Freeze every module represented in inference provenance before the full run.
    $snapshot=Join-Path $stableOutput 'code'
    New-Item -ItemType Directory -Force $snapshot | Out-Null
    Get-ChildItem bench -Filter 'quality_*.py' -File | Copy-Item -Destination $snapshot
    Copy-Item -LiteralPath 'bench/vision.py' -Destination $snapshot
    $stableArgs=@('pose','--bundle','.cache/model-quality/inputs/keyboard','--masks','.cache/model-quality/results/keyboard/segmentation',
        '--device','cuda','--mode','complete','--unlit-templates','--disable-multisampling')
    Write-Output 'Starting keyboard prefix 30'
    & $stablePython -B -m bench.quality_runner @stableArgs --limit 30 --output "$stableOutput/prefix-30.json" *> "$stableOutput/prefix-30.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$stableOutput/prefix-30.log" -Tail 15; throw 'Prefix failed' }
    & $stablePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Prefix viewer update failed' }
    Write-Output 'Starting keyboard full original 240 frames'
    & $stablePython -B -m bench.quality_runner @stableArgs --output "$stableOutput/complete.json" *> "$stableOutput/full.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$stableOutput/full.log" -Tail 15; throw 'Full run failed' }
    & $stablePython -B -m bench.quality_stability_report
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
    & $stablePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Completed viewer update failed' }
} finally { Pop-Location }
