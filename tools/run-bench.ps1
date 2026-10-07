param(
    [string]$UnityEditorRoot = 'C:\Program Files\Unity\Hub\Editor\2022.3.35f1\Editor',
    [string]$Python = 'python',
    [int]$Frames = 24,
    [ValidateSet('plane','box','cylinder')][string]$Shape = 'cylinder'
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$benchOutput = Join-Path $workspace 'artifacts/bench'
New-Item -ItemType Directory -Force $benchOutput | Out-Null
Push-Location $workspace
try {
    & (Join-Path $PSScriptRoot 'export-geometry.ps1') -UnityEditorRoot $UnityEditorRoot
    & $Python -m unittest bench.test_renderer
    if ($LASTEXITCODE -ne 0) { throw 'Reference renderer tests failed.' }
    & $Python -m unittest discover -s tools -p test_evaluate_tracking.py
    if ($LASTEXITCODE -ne 0) { throw 'Attachment scorer tests failed.' }
    & $Python -m bench.run --output $benchOutput --frames $Frames --shape $Shape | Out-File -Encoding utf8 (Join-Path $benchOutput 'run-output.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Simulation failed.' }
    & $Python -m bench.plot_report --output $benchOutput
    if ($LASTEXITCODE -ne 0) { throw 'Plot failed. The selected Python needs the pinned bench requirements.' }
    & $Python -m bench.build_report --output $benchOutput
    if ($LASTEXITCODE -ne 0) { throw 'Interactive report generation failed.' }
    Write-Output "Synthetic mapping model built: $benchOutput/index.html"
    Write-Output 'Known poses and geometry are inputs. No physical AR gate is passed by this model.'
} finally { Pop-Location }
