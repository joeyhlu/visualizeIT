param(
    [ValidateSet('keyboard','mug','ranch')][string[]]$Objects = @('keyboard','mug','ranch'),
    [ValidateSet('controlled','complete')][string[]]$Modes = @('controlled','complete'),
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$qualityRoot = Split-Path -Parent $PSScriptRoot
$qualityPython = Join-Path $qualityRoot '.cache\quality-windows\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $qualityPython)) { throw 'Bootstrap the isolated Windows runtime first.' }
Push-Location -LiteralPath $qualityRoot
function Invoke-QualityStage([string]$Stage, [string[]]$Arguments) {
    $qualityLog = Join-Path $qualityOutput "$Stage-$Device.log"
    & $qualityPython -B -m bench.quality_runner $Stage --bundle $qualityBundle --device $Device @Arguments *> $qualityLog
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $qualityLog -Tail 40
        throw "Quality stage $Stage failed. Preserve this error; retry the same models on CPU if CUDA failed."
    }
}
try {
    foreach ($qualityObject in $Objects) {
        $qualityBundle = ".cache/model-quality/inputs/$qualityObject"
        $qualityOutput = ".cache/model-quality/results/$qualityObject"
        New-Item -ItemType Directory -Force $qualityOutput | Out-Null
        Invoke-QualityStage 'smoke' @('--output',".cache/model-quality/smoke/$qualityObject-$Device.json")
        $qualityBank = ".cache/model-quality/banks/$qualityObject-foundpose.pt"
        if (-not (Test-Path -LiteralPath $qualityBank)) { Invoke-QualityStage 'banks' @('--output',$qualityBank) }
        Invoke-QualityStage 'cnos-bank' @('--output',".cache/model-quality/banks/$qualityObject-cnos.npz")
        Invoke-QualityStage 'segment' @('--output',"$qualityOutput/segmentation")
        foreach ($qualityMode in $Modes) {
            Invoke-QualityStage 'pose' @('--masks',"$qualityOutput/segmentation",'--output',"$qualityOutput/$qualityMode.json",'--mode',$qualityMode)
        }
        & $qualityPython -B -m bench.quality_player
        if ($LASTEXITCODE -ne 0) { throw 'Viewer export failed' }
    }
    & $qualityPython -B -m bench.quality_report
    if ($LASTEXITCODE -ne 0) { throw 'Measured report failed' }
} finally { Pop-Location }
