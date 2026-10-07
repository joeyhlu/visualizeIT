param([switch]$MaskAssociation)
$ErrorActionPreference='Stop'
$recoveryRoot=Split-Path -Parent $PSScriptRoot
$recoveryPython=Join-Path $recoveryRoot '.cache/quality-windows/Scripts/python.exe'
Push-Location -LiteralPath $recoveryRoot
try {
    $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
    $env:NUMEXPR_NUM_THREADS='1'; $env:PYTHONHASHSEED='0'
    Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue
    $activeModels=Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_runner'
    }
    if ($activeModels) { throw 'Another neural experiment is active.' }
    $recoveryOutput=if ($MaskAssociation) { '.cache/model-quality/results/keyboard/recovery-association-1239' } else { '.cache/model-quality/results/keyboard/recovery-1239' }
    if (Test-Path "$recoveryOutput/segmentation/results.json") { throw 'Preserve existing recovery results; choose a new experiment directory.' }
    New-Item -ItemType Directory -Force $recoveryOutput | Out-Null
    $snapshot=Join-Path $recoveryOutput 'code'
    New-Item -ItemType Directory -Force $snapshot | Out-Null
    Get-ChildItem bench -Filter 'quality_*.py' -File | Copy-Item -Destination $snapshot
    Copy-Item -LiteralPath 'bench/vision.py' -Destination $snapshot
    $recoveryCommon=@('--bundle','.cache/model-quality/inputs/keyboard','--device','cuda')
    function Invoke-RecoveryStage([string]$Name,[string[]]$Arguments) {
        Write-Output "Starting recovery $Name"
        & $recoveryPython -B -m bench.quality_runner @Arguments *> "$recoveryOutput/$Name.log"
        if ($LASTEXITCODE -ne 0) {
            Get-Content "$recoveryOutput/$Name.log" -Tail 20
            throw "Recovery stage $Name failed; preserve diagnostics."
        }
    }
    if (-not (Test-Path '.cache/model-quality/banks/keyboard-cnos.npz')) {
        Invoke-RecoveryStage 'cnos-bank' (@('cnos-bank')+$recoveryCommon+@('--output','.cache/model-quality/banks/keyboard-cnos.npz'))
    }
    $associationArgs=if ($MaskAssociation) { @('--mask-association') } else { @() }
    Invoke-RecoveryStage 'segmentation' (@('segment')+$recoveryCommon+$associationArgs+@('--force-occlusion','1239','--output',"$recoveryOutput/segmentation"))
    Invoke-RecoveryStage 'pose' (@('pose')+$recoveryCommon+@('--masks',"$recoveryOutput/segmentation",'--mode','complete',
        '--unlit-templates','--disable-multisampling','--force-occlusion','1239','--output',"$recoveryOutput/complete.json"))
    & $recoveryPython -B -m bench.quality_recovery_report --root $recoveryOutput
    if ($LASTEXITCODE -ne 0) { throw 'Recovery evaluation failed' }
    & $recoveryPython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Recovery viewer publication failed' }
} finally { Pop-Location }
