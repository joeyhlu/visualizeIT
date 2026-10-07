param()
$ErrorActionPreference = 'Stop'
$stabilityRoot = Split-Path -Parent $PSScriptRoot
$stabilityPython = Join-Path $stabilityRoot '.cache/quality-windows/Scripts/python.exe'
Push-Location -LiteralPath $stabilityRoot
try {
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:PYTHONHASHSEED = '0'
    $activeModels = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_runner'
    }
    if ($activeModels) { throw 'Another neural experiment is active.' }
    $stabilityOutput = '.cache/model-quality/results/keyboard/render-stability'
    New-Item -ItemType Directory -Force $stabilityOutput | Out-Null
    if (Test-Path "$stabilityOutput/prefix-2.json") { throw 'Preserve previous experiment; use a new output folder.' }
    function Invoke-StabilityStage([string]$Name, [string[]]$Arguments) {
        Write-Output "Starting $Name"
        if ($Name -in @('prefix-2','prefix-4')) { $env:VISUALIZEIT_QUALITY_TRACE = "$stabilityOutput/$Name-trace.jsonl" }
        else { Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue }
        & $stabilityPython -B -m bench.quality_runner @Arguments *> "$stabilityOutput/$Name.log"
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath "$stabilityOutput/$Name.log" -Tail 30
            throw "Render stability experiment failed at $Name"
        }
    }
    $stabilityCommon = @('--bundle','.cache/model-quality/inputs/keyboard','--device','cuda','--unlit-templates','--disable-multisampling')
    Invoke-StabilityStage 'smoke' (@('smoke')+$stabilityCommon+@('--output','.cache/model-quality/smoke/keyboard-cuda-unlit-no-msaa.json'))
    $stabilityPose = @('pose')+$stabilityCommon+@('--masks','.cache/model-quality/results/keyboard/segmentation','--mode','complete')
    Invoke-StabilityStage 'prefix-2' ($stabilityPose+@('--limit','2','--output',"$stabilityOutput/prefix-2.json"))
    Invoke-StabilityStage 'prefix-4' ($stabilityPose+@('--limit','4','--output',"$stabilityOutput/prefix-4.json"))
    & $stabilityPython -B -c "import json;from pathlib import Path;from bench.quality_evaluate import prefix_equal;from bench.quality_assets import save_result;p=Path('$stabilityOutput');a=json.loads((p/'prefix-2.json').read_text());b=json.loads((p/'prefix-4.json').read_text());r=dict(same_provenance=a['provenance']==b['provenance'],prefix_identical=prefix_equal(a,b),scope='Two versus four original keyboard frames; full prefix gate still required');save_result(p/'report.json',r);print(r)"
    if ($LASTEXITCODE -ne 0) { throw 'Prefix comparison failed' }
    & $stabilityPython -B -m bench.quality_trace_compare --short "$stabilityOutput/prefix-2-trace.jsonl" --long "$stabilityOutput/prefix-4-trace.jsonl" --output "$stabilityOutput/trace-comparison.json"
    if ($LASTEXITCODE -ne 0) { throw 'Trace comparison failed' }
} finally { Pop-Location }
