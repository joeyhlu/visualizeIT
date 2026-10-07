param(
    [string]$Python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [ValidateSet('All','Verify','Mapping','Acquire','Control','Track','Evaluate','Interruptions','HD','Performance','HigherFPS','SixtyFPS','Recovery','LearnedMask','RecoveryPlayer')]
    [string]$Stage = 'All'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with a Python 3.12 executable containing bench/requirements-real.txt.' }
function Invoke-Offline([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Offline command failed: $($Arguments -join ' ')" }
}
Push-Location -LiteralPath $taskRoot
try {
    if ($Stage -in @('All','Verify')) {
        Invoke-Offline @('-B','-m','unittest','bench.test_renderer','bench.test_scan_mapping','bench.test_improvements','bench.test_remote_zip','bench.test_hd_experiment','bench.test_show3d','bench.test_surface_tracker','bench.test_visible_mask')
        Invoke-Offline @('tools/test_evaluate_tracking.py')
    }
    if ($Stage -in @('All','Mapping')) { Invoke-Offline @('-m','bench.mapping_experiment') }
    if ($Stage -in @('All','Acquire')) { Invoke-Offline @('-m','bench.bop_assets','--include-limited-motion') }
    if ($Stage -eq 'Control') { Invoke-Offline @('-m','bench.video_experiment','--operation','control') }
    if ($Stage -eq 'HD') { Invoke-Offline @('-m','bench.hd_experiment') }
    if ($Stage -eq 'HigherFPS') { Invoke-Offline @('-m','bench.speed_sweep') }
    if ($Stage -eq 'SixtyFPS') { Invoke-Offline @('-B','-m','bench.show3d_experiment') }
    if ($Stage -eq 'Recovery') { Invoke-Offline @('-B','-m','bench.retry_experiment','--mask','--synthetic','--output','artifacts/video-60/retry-model-mask') }
    if ($Stage -eq 'LearnedMask') {
        Invoke-Offline @('-B','-m','bench.segmentation_assets')
        Invoke-Offline @('-B','-m','bench.learned_mask_experiment')
        Invoke-Offline @('-B','-m','bench.retry_experiment','--learned','--synthetic','--output','artifacts/video-60/retry-learned-mask')
    }
    if ($Stage -eq 'RecoveryPlayer') { Invoke-Offline @('-B','-m','bench.recovery_player') }
    if ($Stage -eq 'Performance') {
        Invoke-Offline @('-m','bench.performance_experiment','--height','480','--output','artifacts/video-hd/performance-480')
        Invoke-Offline @('-m','bench.performance_experiment','--height','480','--compact','--output','artifacts/video-hd/performance-480-compact')
    }
    if ($Stage -in @('All','Track')) { Invoke-Offline @('-m','bench.video_experiment','--operation','track') }
    if ($Stage -in @('All','Evaluate')) { Invoke-Offline @('-m','bench.video_experiment','--operation','evaluate') }
    if ($Stage -in @('All','Interruptions')) {
        Invoke-Offline @('-m','bench.video_experiment','--object','15','--interruptions','--render-stride','110','--output','artifacts/video-interruptions')
        Invoke-Offline @('-m','bench.video_experiment','--operation','evaluate','--output','artifacts/video-interruptions')
    }
    Write-Output 'Offline stage complete. Review artifacts/mapping-improvements and artifacts/video-tracking. Phone validation remains pending.'
} finally { Pop-Location }
