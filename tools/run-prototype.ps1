[CmdletBinding(PositionalBinding=$false)]
param(
    [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [Parameter(Mandatory=$true)][string]$Module,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ModuleArgs
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with the pinned Python 3.12 runtime.' }
Push-Location -LiteralPath $taskRoot
try {
    & $Python -B -S tools/prototype_bootstrap.py $Module @ModuleArgs
    if ($LASTEXITCODE -ne 0) { throw "Prototype command exited $LASTEXITCODE" }
} finally { Pop-Location }
