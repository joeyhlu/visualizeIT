param([string]$BootstrapPython = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe")
$ErrorActionPreference = 'Stop'
$qualityRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $qualityRoot
function Invoke-QualityPython([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Quality runtime command failed: $($Arguments -join ' ')" }
}
try {
    $qualityPython = Join-Path $qualityRoot '.cache\quality-windows\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $qualityPython)) {
        Invoke-QualityPython $BootstrapPython @('-m','venv','.cache/quality-windows')
    }
    Invoke-QualityPython $qualityPython @('-m','pip','install','pip==24.3.1','setuptools==75.3.0','wheel==0.44.0')
    Invoke-QualityPython $qualityPython @('-m','pip','install','torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu124','--no-cache-dir')
    Invoke-QualityPython $qualityPython @('-m','pip','install','-r','bench/runtime/requirements-resolved-windows.txt','--extra-index-url','https://download.pytorch.org/whl/cu124','--no-cache-dir')
    Invoke-QualityPython $qualityPython @('-m','pip','check')
    Write-Output "Isolated Windows runtime ready: $qualityPython. Docker was not started."
} finally { Pop-Location }
