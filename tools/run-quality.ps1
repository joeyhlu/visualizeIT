param(
    [ValidateSet('Prepare','Verify','Viewer')][string]$Stage = 'Verify',
    [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)
$ErrorActionPreference = 'Stop'
$qualityRoot = Split-Path -Parent $PSScriptRoot
function Invoke-Quality([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Quality command failed: $($Arguments -join ' ')" }
}
Push-Location -LiteralPath $qualityRoot
try {
    if ($Stage -eq 'Prepare') {
        Invoke-Quality @('-B','-m','bench.quality_assets')
        Invoke-Quality @('-B','-m','bench.quality_dataset')
    }
    if ($Stage -eq 'Verify') {
        & "$PSScriptRoot\run-offline.ps1" -Python $Python -Stage Verify
        Invoke-Quality @('-B','-m','unittest','bench.test_quality')
    }
    if ($Stage -in @('Prepare','Viewer')) {
        Invoke-Quality @('-B','-m','bench.quality_annotations')
        Invoke-Quality @('-B','-m','bench.quality_player')
    }
    Write-Output 'Code checks/preparation complete. No Docker startup or model inference was attempted.'
} finally { Pop-Location }
