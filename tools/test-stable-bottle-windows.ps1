param()
$ErrorActionPreference='Stop'
$bottleRoot=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $bottleRoot
try {
    $bottlePython=Join-Path $bottleRoot '.cache/quality-windows/Scripts/python.exe'
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    $env:NUMEXPR_NUM_THREADS='1'; $env:PYTHONHASHSEED='0'
    Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue
    $bottleActive=Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_(runner|contour_neural_probe)'
    }
    if ($bottleActive) { throw 'Another neural experiment is active.' }
    $bottleOutput='.cache/model-quality/results/ranch/render-stability'
    if (Test-Path $bottleOutput) { throw 'Preserve previous bottle render experiment.' }
    New-Item -ItemType Directory -Force "$bottleOutput/code" | Out-Null
    Get-ChildItem bench -Filter 'quality_*.py' -File | Copy-Item -Destination "$bottleOutput/code"
    Copy-Item -LiteralPath 'bench/vision.py' -Destination "$bottleOutput/code"
    $bottleArgs=@('pose','--bundle','.cache/model-quality/inputs/ranch','--masks','.cache/model-quality/results/ranch/segmentation',
        '--device','cuda','--mode','complete','--unlit-templates','--disable-multisampling')
    Write-Output 'Starting bottle renderer/checkpoint smoke'
    & $bottlePython -B -m bench.quality_runner smoke --bundle '.cache/model-quality/inputs/ranch' --device cuda --unlit-templates --disable-multisampling --output '.cache/model-quality/smoke/ranch-cuda-unlit-no-msaa.json' *> "$bottleOutput/smoke.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$bottleOutput/smoke.log" -Tail 15; throw 'Bottle smoke failed' }
    Write-Output 'Starting bottle prefix 30'
    & $bottlePython -B -m bench.quality_runner @bottleArgs --limit 30 --output "$bottleOutput/prefix-30.json" *> "$bottleOutput/prefix-30.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$bottleOutput/prefix-30.log" -Tail 15; throw 'Bottle prefix failed' }
    & $bottlePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Bottle prefix publication failed' }
    Write-Output 'Starting bottle full original 240 frames'
    & $bottlePython -B -m bench.quality_runner @bottleArgs --output "$bottleOutput/complete.json" *> "$bottleOutput/full.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$bottleOutput/full.log" -Tail 15; throw 'Bottle full run failed' }
    & $bottlePython -B -m bench.quality_bottle_stability_report
    if ($LASTEXITCODE -ne 0) { throw 'Bottle evaluation failed' }
    & $bottlePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Bottle full publication failed' }
} finally { Pop-Location }
