param([string]$UnityEditorRoot = 'C:\Program Files\Unity\Hub\Editor\2022.3.35f1\Editor')
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$shaderProject = Join-Path $workspace '.cache/shader-check'
New-Item -ItemType Directory -Force (Join-Path $shaderProject 'Assets/Editor'),(Join-Path $shaderProject 'Packages'),(Join-Path $shaderProject 'ProjectSettings') | Out-Null
$editorVersion = (Get-Item (Join-Path $UnityEditorRoot 'Unity.exe')).VersionInfo.ProductVersion
if ($editorVersion -notmatch '^2022\.3\.') { throw 'This supplemental harness expects the installed Unity 2022.3 editor, not the target mobile project.' }
'{"dependencies":{"com.unity.render-pipelines.universal":"14.0.11"}}' | Set-Content (Join-Path $shaderProject 'Packages/manifest.json')
"m_EditorVersion: $editorVersion" | Set-Content (Join-Path $shaderProject 'ProjectSettings/ProjectVersion.txt')
Copy-Item -LiteralPath (Join-Path $workspace 'mobile/Assets/VisualizeIt/Resources/SurfacePattern.shader') -Destination (Join-Path $shaderProject 'Assets/SurfacePattern.shader') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'ShaderCheck.cs') -Destination (Join-Path $shaderProject 'Assets/Editor/ShaderCheck.cs') -Force
$log = Join-Path $shaderProject 'shader-check.log'
$editorProcess = Start-Process -FilePath (Join-Path $UnityEditorRoot 'Unity.exe') -ArgumentList @('-batchmode','-projectPath',('"' + $shaderProject + '"'),'-executeMethod','ShaderCheck.Run','-logFile',('"' + $log + '"')) -WindowStyle Hidden -PassThru
$editorProcess.WaitForExit()
if ($editorProcess.ExitCode -ne 0) { throw "Supplemental shader check failed. See $log" }
Write-Output "Supplemental shader compilation passed in Unity 2022.3. This does not replace Unity 6.3 or phone validation. Log: $log"
