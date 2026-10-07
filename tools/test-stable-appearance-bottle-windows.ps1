param()
$ErrorActionPreference='Stop'
$appearanceRoot=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $appearanceRoot
try {
    $appearancePython=Join-Path $appearanceRoot '.cache/quality-windows/Scripts/python.exe'
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    $env:NUMEXPR_NUM_THREADS='1'; $env:PYTHONHASHSEED='0'
    Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue
    $appearanceActive=Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_(runner|contour_neural_probe|identity_current_probe)'
    }
    if ($appearanceActive) { throw 'Another model experiment is active.' }
    $appearanceOutput='.cache/model-quality/results/ranch/render-stability-appearance'
    if (Test-Path $appearanceOutput) { throw 'Preserve previous stable appearance results.' }
    New-Item -ItemType Directory -Force "$appearanceOutput/code" | Out-Null
    Get-ChildItem bench -Filter 'quality_*.py' -File | Copy-Item -Destination "$appearanceOutput/code"
    Copy-Item -LiteralPath 'bench/vision.py' -Destination "$appearanceOutput/code"
    $appearanceArgs=@('pose','--bundle','.cache/model-quality/inputs/ranch','--masks','.cache/model-quality/results/ranch/segmentation',
        '--device','cuda','--mode','complete','--unlit-templates','--disable-multisampling','--appearance-check')
    Write-Output 'Starting stable appearance bottle prefix 30'
    & $appearancePython -B -m bench.quality_runner @appearanceArgs --limit 30 --output "$appearanceOutput/prefix-30.json" *> "$appearanceOutput/prefix-30.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$appearanceOutput/prefix-30.log" -Tail 15; throw 'Appearance prefix failed' }
    & $appearancePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Appearance prefix publication failed' }
    Write-Output 'Starting stable appearance bottle full original 240 frames'
    & $appearancePython -B -m bench.quality_runner @appearanceArgs --output "$appearanceOutput/complete.json" *> "$appearanceOutput/full.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$appearanceOutput/full.log" -Tail 15; throw 'Appearance full run failed' }
    & $appearancePython -B -m bench.quality_bottle_stable_appearance_report
    if ($LASTEXITCODE -ne 0) { throw 'Appearance full evaluation failed' }
    & $appearancePython -B -m bench.quality_report
    if ($LASTEXITCODE -ne 0) { throw 'Measured summaries publication failed' }
    & $appearancePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Appearance full publication failed' }
} finally { Pop-Location }
