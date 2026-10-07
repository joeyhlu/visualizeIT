param(
    [string]$UnityEditorRoot = 'C:\Program Files\Unity\Hub\Editor\2022.3.35f1\Editor',
    [string]$Python = 'python',
    [switch]$Plot
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$exportOutput = Join-Path $workspace 'artifacts/geometry'
$compilerOutput = Join-Path $workspace '.cache/verification'
New-Item -ItemType Directory -Force $exportOutput,$compilerOutput | Out-Null
$monoRoot = Join-Path $UnityEditorRoot 'Data/MonoBleedingEdge'
$mono = Join-Path $monoRoot 'bin/mono.exe'
$compiler = Join-Path $monoRoot 'lib/mono/4.5/csc.exe'
$exporter = Join-Path $compilerOutput 'ExportGeometry.exe'
& $mono $compiler -nologo -langversion:latest "-out:$exporter" (Join-Path $workspace 'mobile/Assets/VisualizeIt/Core/SurfaceGeometry.cs') (Join-Path $workspace 'mobile/Assets/VisualizeIt/Core/PatternMapping.cs') (Join-Path $workspace 'mobile/Assets/VisualizeIt/Core/MeshMappingAnalysis.cs') (Join-Path $PSScriptRoot 'ExportGeometry.cs')
if ($LASTEXITCODE -ne 0) { throw 'Geometry exporter compilation failed.' }
& $mono $exporter $exportOutput
if ($LASTEXITCODE -ne 0) { throw 'Geometry export failed.' }
if ($Plot) {
    & $Python (Join-Path $PSScriptRoot 'geometry_report.py')
    if ($LASTEXITCODE -ne 0) { throw 'Plot failed. The chosen Python needs NumPy and Matplotlib.' }
}
