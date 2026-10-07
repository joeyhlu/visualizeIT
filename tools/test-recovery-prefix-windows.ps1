param()
$ErrorActionPreference='Stop'
$prefixRoot=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $prefixRoot
try {
    $prefixPython=Join-Path $prefixRoot '.cache/quality-windows/Scripts/python.exe'
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    $env:NUMEXPR_NUM_THREADS='1'; $env:PYTHONHASHSEED='0'
    Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue
    $activeModels=Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_runner'
    }
    if ($activeModels) { throw 'Another neural experiment is active.' }
    $prefixOutput='.cache/model-quality/results/keyboard/recovery-association-prefix-120'
    if (Test-Path $prefixOutput) { throw 'Preserve previous prefix experiment.' }
    New-Item -ItemType Directory -Force "$prefixOutput/code" | Out-Null
    Get-ChildItem bench -Filter 'quality_*.py' -File | Copy-Item -Destination "$prefixOutput/code"
    Copy-Item -LiteralPath 'bench/vision.py' -Destination "$prefixOutput/code"
    function Invoke-PrefixStage([string]$Name,[string[]]$Arguments) {
        Write-Output "Starting recovery prefix $Name"
        & $prefixPython -B -m bench.quality_runner @Arguments *> "$prefixOutput/$Name.log"
        if ($LASTEXITCODE -ne 0) { Get-Content "$prefixOutput/$Name.log" -Tail 20; throw "Prefix stage $Name failed." }
    }
    $prefixCommon=@('--bundle','.cache/model-quality/inputs/keyboard','--device','cuda','--limit','120','--force-occlusion','1239')
    Invoke-PrefixStage 'segmentation' (@('segment')+$prefixCommon+@('--mask-association','--output',"$prefixOutput/segmentation"))
    Invoke-PrefixStage 'pose' (@('pose')+$prefixCommon+@('--masks',"$prefixOutput/segmentation",'--mode','complete',
        '--unlit-templates','--disable-multisampling','--output',"$prefixOutput/complete.json"))
    & $prefixPython -B -m bench.quality_recovery_prefix_report
    if ($LASTEXITCODE -ne 0) { throw 'Recovery prefix evaluation failed.' }
    & $prefixPython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Recovery prefix viewer publication failed.' }
} finally { Pop-Location }
