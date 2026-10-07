param(
    [string]$UnityEditorRoot = 'C:\Program Files\Unity\Hub\Editor\2022.3.35f1\Editor',
    [string]$Cxx = 'g++'
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$verificationOutput = Join-Path $workspace '.cache/verification'
New-Item -ItemType Directory -Force $verificationOutput | Out-Null
$monoRoot = Join-Path $UnityEditorRoot 'Data/MonoBleedingEdge'
$mono = Join-Path $monoRoot 'bin/mono.exe'
$compiler = Join-Path $monoRoot 'lib/mono/4.5/csc.exe'
if (!(Test-Path $mono) -or !(Test-Path $compiler)) { throw 'Pass -UnityEditorRoot with an installed Unity editor containing Mono/Roslyn.' }
$core = Join-Path $workspace 'mobile/Assets/VisualizeIt/Core'
$testExe = Join-Path $verificationOutput 'CoreTests.exe'
& $mono $compiler -nologo -langversion:latest "-out:$testExe" (Join-Path $core 'SurfaceGeometry.cs') (Join-Path $core 'TrackingVisibility.cs') (Join-Path $core 'PatternMapping.cs') (Join-Path $core 'MeshMappingAnalysis.cs') (Join-Path $workspace 'mobile/Assets/VisualizeIt/Tests/CoreVerification.cs') (Join-Path $PSScriptRoot 'CoreTestRunner.cs')
if ($LASTEXITCODE -ne 0) { throw 'Core test compilation failed.' }
& $mono $testExe
if ($LASTEXITCODE -ne 0) { throw 'Core checks failed.' }
$roslyn = Join-Path $monoRoot 'lib/mono/4.5'
$references = @('Microsoft.CodeAnalysis.dll','Microsoft.CodeAnalysis.CSharp.dll','System.Collections.Immutable.dll','System.Reflection.Metadata.dll')
$referenceArguments = @()
foreach ($assembly in $references) {
    $assemblyPath = Join-Path $roslyn $assembly
    Copy-Item -LiteralPath $assemblyPath -Destination $verificationOutput -Force
    $referenceArguments += "-r:$assemblyPath"
}
foreach ($dependency in @('System.Memory.dll','System.Buffers.dll','System.Runtime.CompilerServices.Unsafe.dll','System.Threading.Tasks.Extensions.dll')) {
    $dependencyPath = Join-Path $roslyn $dependency
    if (Test-Path $dependencyPath) { Copy-Item -LiteralPath $dependencyPath -Destination $verificationOutput -Force }
}
$referenceArguments += ('-r:' + (Join-Path $roslyn 'Facades/netstandard.dll'))
$syntaxExe = Join-Path $verificationOutput 'SyntaxCheck.exe'
& $mono $compiler -nologo -langversion:latest "-out:$syntaxExe" $referenceArguments (Join-Path $PSScriptRoot 'SyntaxCheck.cs')
if ($LASTEXITCODE -ne 0) { throw 'Syntax validator compilation failed.' }
& $mono $syntaxExe (Join-Path $workspace 'mobile/Assets/VisualizeIt')
if ($LASTEXITCODE -ne 0) { throw 'C# syntax checks failed.' }
$nativeExe = Join-Path $verificationOutput 'pose_tests.exe'
& $Cxx -std=c++17 -Wall -Wextra -Werror '-I' (Join-Path $workspace 'native/include') (Join-Path $workspace 'native/src/pose.cpp') (Join-Path $workspace 'native/tests/pose_tests.cpp') -o $nativeExe
if ($LASTEXITCODE -ne 0) { throw 'Native test compilation failed.' }
& $nativeExe
if ($LASTEXITCODE -ne 0) { throw 'Native coordinate checks failed.' }
Write-Output 'Local source checks passed. Unity 6 compilation and physical AR validation remain required.'
