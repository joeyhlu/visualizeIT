<#
.SYNOPSIS
  Start the local VisualizeIT video validation workbench (127.0.0.1 only).
.DESCRIPTION
  Uses an existing Python environment; installs nothing and downloads nothing (models are never
  fetched by this launcher). Neural/segmentation work runs only when explicitly triggered in the
  UI/CLI. Every argument is passed to Python as its own native argument (no command-string
  composition). The service remains authoritative: it re-validates every asset, bundle,
  execution control and the serving admission itself.

  -ReviewedSixCase uses the reviewed local six-case configuration (pinned GPU Python, the
  flow-terminal head, admission attempt-08, three historical bundles, three execution-only
  supplemental controls, six mask sets, COLMAP 3.11.1, device cuda). Missing assets fail; nothing is
  substituted. Explicit asset/geometry/mask/device/python overrides are rejected with the preset.

  -Plan prints the exact executable and argument arrays as JSON and exits without starting the
  server, segmentation or inference. It is not approval and does not weaken the service's checks.
.EXAMPLE
  ./tools/run-workbench.ps1 -ReviewedSixCase
  ./tools/run-workbench.ps1 -ReviewedSixCase -Plan
  ./tools/run-workbench.ps1 -Port 8765
  ./tools/run-workbench.ps1 -MetricBundleFor dev-mug=C:\x\bundle-mug.json -MasksFor dev-mug=sam2-workbench-01
  ./tools/run-workbench.ps1 -Root mymedia=D:\clips -Colmap C:\COLMAP\COLMAP.bat
#>
param(
  [int]$Port = 8765,
  [string]$Workspace = "",
  [string[]]$Root = @(),
  [string]$Colmap = "",
  [string]$Head = "",
  [string]$Admission = "",
  [string]$MetricBundle = "",
  [string]$MetricBundleClip = "",
  [string[]]$MetricBundleFor = @(),
  [string[]]$ExecutionControlFor = @(),
  [string[]]$MasksFor = @(),
  [string]$Masks = "",
  [string]$Device = "cpu",
  [string]$Python = "python",
  [switch]$ReviewedSixCase,
  [switch]$Plan,
  [switch]$ExportSuite,
  [switch]$Inspect
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$ExpectedHeadSha256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
$ClipIdPattern = '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$'

function Fail([string]$Message) {
  [Console]::Error.WriteLine("[workbench] error: $Message")
  exit 2
}

function Get-Sha256Hex([string]$Path) {
  $stream = [System.IO.File]::OpenRead($Path)
  try {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return (-join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("x2") })) }
    finally { $sha.Dispose() }
  } finally { $stream.Dispose() }
}

# Parse CLIP_ID=VALUE items; reject malformed entries and duplicate clip ids within one option.
# Items are also split on commas, because `powershell -File` passes `-X a=1,b=2` as ONE string.
# A value containing a comma therefore fails as malformed instead of being silently misread.
function Get-ClipMap([string[]]$Items, [string]$Option) {
  $map = [ordered]@{}
  foreach ($item in @($Items | ForEach-Object { $_ -split "," })) {
    $index = $item.IndexOf("=")
    if ($index -lt 1 -or $index -eq $item.Length - 1) { Fail "$Option expects CLIP_ID=VALUE, got '$item'" }
    $clip = $item.Substring(0, $index)
    $value = $item.Substring($index + 1)
    if ($clip -notmatch $ClipIdPattern) { Fail "$Option has an invalid clip id '$clip'" }
    if ($map.Contains($clip)) { Fail "$Option maps clip '$clip' more than once" }
    $map[$clip] = $value
  }
  return $map
}

if ($Port -lt 1 -or $Port -gt 65535) { Fail "port must be within 1..65535" }

if ($ReviewedSixCase) {
  $presetOwned = @("Python", "Head", "Admission", "Colmap", "Device", "MetricBundle", "MetricBundleClip",
                   "MetricBundleFor", "ExecutionControlFor", "MasksFor", "Masks", "Workspace")
  $conflicts = @($presetOwned | Where-Object { $PSBoundParameters.ContainsKey($_) })
  if ($conflicts.Count -gt 0) {
    Fail ("-ReviewedSixCase fixes these settings; remove the explicit override(s) or use the generic mode: " +
          ($conflicts -join ", "))
  }
  $Workspace = Join-Path $repo ".cache\video-workbench"
  $Python = Join-Path $repo ".cache\quality-windows\Scripts\python.exe"
  $Head = Join-Path $repo ".cache\video-workbench\models\flow-terminal-20261021\head.pt"
  $Admission = Join-Path $repo ".cache\video-workbench\admission\attempt-08"
  $Colmap = Join-Path $repo ".cache\video-workbench-tools\colmap-3.11.1\bin\colmap.exe"
  $Device = "cuda"
  $adm = Join-Path $repo ".cache\video-workbench\admission"
  $ctl = Join-Path $repo ".cache\video-workbench\derived\execution-controls\v1"
  $MetricBundleFor = @("dev-mug=$(Join-Path $adm 'bundle-mug-v1.json')",
                       "dev-keyboard=$(Join-Path $adm 'bundle-keyboard-v1.json')",
                       "dev-ranch=$(Join-Path $adm 'bundle-ranch-v1.json')")
  $ExecutionControlFor = @("dev-ycb-003-cracker-box=$(Join-Path $ctl 'object-000002')",
                           "dev-ycb-005-tomato-soup-can=$(Join-Path $ctl 'object-000004')",
                           "dev-ycb-040-large-marker=$(Join-Path $ctl 'object-000018')")
  $MasksFor = @("dev-mug=sam2-workbench-01", "dev-keyboard=sam2-workbench-02", "dev-ranch=sam2-workbench-01",
                "dev-ycb-003-cracker-box=sam2-workbench-01", "dev-ycb-005-tomato-soup-can=sam2-workbench-01",
                "dev-ycb-040-large-marker=sam2-workbench-01")
}
if (-not $Workspace) { $Workspace = Join-Path $repo ".cache\video-workbench" }

$bundleMap = Get-ClipMap $MetricBundleFor "-MetricBundleFor"
$controlMap = Get-ClipMap $ExecutionControlFor "-ExecutionControlFor"
$maskMap = Get-ClipMap $MasksFor "-MasksFor"
foreach ($clip in $controlMap.Keys) {
  if ($bundleMap.Contains($clip)) { Fail "clip '$clip' has both a metric bundle and an execution control; configure exactly one geometry" }
}
if ($MetricBundleClip) {
  if ($MetricBundleClip -notmatch $ClipIdPattern) { Fail "-MetricBundleClip has an invalid clip id '$MetricBundleClip'" }
  if (-not $MetricBundle) { Fail "-MetricBundleClip requires -MetricBundle" }
  if ($bundleMap.Contains($MetricBundleClip) -or $controlMap.Contains($MetricBundleClip)) {
    Fail "clip '$MetricBundleClip' is bound by -MetricBundleClip and also has a per-clip geometry mapping"
  }
}
foreach ($mask in $maskMap.Values) {
  if ($mask -notmatch $ClipIdPattern) { Fail "-MasksFor has an invalid mask set name '$mask'" }
}

if ($ReviewedSixCase) {
  # Presence check of every preset asset (no substitution), then the pinned head identity.
  $missing = @()
  foreach ($file in @($Python, $Head, $Colmap) + @($bundleMap.Values)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { $missing += $file }
  }
  foreach ($dir in @($Admission) + @($controlMap.Values)) {
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { $missing += $dir }
  }
  foreach ($dir in $controlMap.Values) {
    foreach ($name in @("model.glb", "metric-bundle.json", "execution-control-sidecar.json", "emission-receipt.json")) {
      $path = Join-Path $dir $name
      if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $missing += $path }
    }
  }
  foreach ($clip in $maskMap.Keys) {
    $manifest = Join-Path $Workspace (Join-Path "masks\$clip" (Join-Path $maskMap[$clip] "manifest.json"))
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { $missing += $manifest }
  }
  if ($missing.Count -gt 0) {
    Fail ("-ReviewedSixCase assets are missing (the launcher never downloads or substitutes them):`n  " +
          ($missing -join "`n  "))
  }
  $headSha = Get-Sha256Hex $Head
  if ($headSha -ne $ExpectedHeadSha256) {
    Fail "head.pt SHA-256 $headSha differs from the pinned $ExpectedHeadSha256; refusing to start"
  }
}

$commands = @()
$common = @("-m", "bench.workbench_cli", "--workspace", $Workspace)
if ($Inspect) {
  $cliArgs = $common + @("inspect")
  if ($Colmap) { $cliArgs += @("--colmap", $Colmap) }
  if ($Head) { $cliArgs += @("--head", $Head) }
  $commands += , [pscustomobject]@{ purpose = "inspect"; executable = $Python; arguments = [string[]]$cliArgs }
} else {
  if ($ExportSuite) {
    $commands += , [pscustomobject]@{ purpose = "export_suite"; executable = $Python;
                                       arguments = [string[]]($common + @("suite")) }
  }
  $serve = @("-m", "bench.workbench_service", "--workspace", $Workspace, "--host", "127.0.0.1", "--port", "$Port",
             "--device", $Device)
  foreach ($r in $Root) { $serve += @("--root", $r) }
  if ($Colmap) { $serve += @("--colmap", $Colmap) }
  if ($Head) { $serve += @("--head", $Head) }
  if ($Admission) { $serve += @("--admission", $Admission) }
  if ($MetricBundle) { $serve += @("--metric-bundle", $MetricBundle) }
  if ($MetricBundleClip) { $serve += @("--metric-bundle-clip", $MetricBundleClip) }
  foreach ($clip in $bundleMap.Keys) { $serve += @("--metric-bundle-for", "$clip=$($bundleMap[$clip])") }
  foreach ($clip in $controlMap.Keys) { $serve += @("--execution-control-for", "$clip=$($controlMap[$clip])") }
  if ($Masks) { $serve += @("--masks", $Masks) }
  foreach ($clip in $maskMap.Keys) { $serve += @("--masks-for", "$clip=$($maskMap[$clip])") }
  $commands += , [pscustomobject]@{ purpose = "serve"; executable = $Python; arguments = [string[]]$serve }
}

if ($Plan) {
  $planReport = [pscustomobject]@{
    mode = "plan"
    preset = $(if ($ReviewedSixCase) { "reviewed_six_case" } else { $null })
    head_sha256_verified = $(if ($ReviewedSixCase) { $ExpectedHeadSha256 } else { $null })
    loopback = "127.0.0.1"
    commands = $commands
    note = "dry run only: nothing was started; not approval; the service re-validates every asset and the admission"
  }
  Write-Output (ConvertTo-Json -InputObject $planReport -Depth 6)
  exit 0
}

foreach ($command in $commands) {
  if ($command.purpose -eq "serve") {
    Write-Host "[workbench] serving $Workspace on http://127.0.0.1:$Port/ (Ctrl+C to stop)"
  }
  $arguments = $command.arguments
  & $command.executable @arguments
  if ($LASTEXITCODE -ne 0 -or $command.purpose -ne "export_suite") { exit $LASTEXITCODE }
}
