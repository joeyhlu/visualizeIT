param()
$ErrorActionPreference = 'Stop'
$appearanceRoot = Split-Path -Parent $PSScriptRoot
$appearancePython = Join-Path $appearanceRoot '.cache/quality-windows/Scripts/python.exe'
Push-Location -LiteralPath $appearanceRoot
try {
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:PYTHONHASHSEED = '0'
    # Invoke only after the previous neural GPU experiment has finished.
    $activeModels = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_runner'
    }
    if ($activeModels) { throw 'Another neural quality stage is active; preserve its session and wait.' }
    $appearanceOutput = '.cache/model-quality/results/ranch/appearance'
    New-Item -ItemType Directory -Force $appearanceOutput | Out-Null
    function Invoke-AppearanceStage([string]$Name, [string[]]$Arguments) {
        Write-Output "Starting $Name"
        if ($Name -in @('repeat-short','repeat-long')) { $env:VISUALIZEIT_QUALITY_TRACE = "$appearanceOutput/$Name-trace.jsonl" }
        else { Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue }
        & $appearancePython -B -m bench.quality_runner @Arguments *> "$appearanceOutput/$Name.log"
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath "$appearanceOutput/$Name.log" -Tail 30
            throw "Appearance experiment failed at $Name; preserve its output."
        }
    }
    $appearanceCommon = @('--bundle','.cache/model-quality/inputs/ranch','--device','cuda','--unlit-templates')
    Invoke-AppearanceStage 'smoke-bottle' (@('smoke')+$appearanceCommon+@('--output','.cache/model-quality/smoke/ranch-cuda-unlit.json'))
    $keyboardCommon = @('--bundle','.cache/model-quality/inputs/keyboard','--device','cuda','--unlit-templates')
    Invoke-AppearanceStage 'smoke-keyboard' (@('smoke')+$keyboardCommon+@('--output','.cache/model-quality/smoke/keyboard-cuda-unlit.json'))
    Invoke-AppearanceStage 'repeat-short' (@('pose')+$keyboardCommon+@('--masks','.cache/model-quality/results/keyboard/segmentation','--mode','complete','--limit','2','--output',"$appearanceOutput/keyboard-prefix-2.json"))
    Invoke-AppearanceStage 'repeat-long' (@('pose')+$keyboardCommon+@('--masks','.cache/model-quality/results/keyboard/segmentation','--mode','complete','--limit','4','--output',"$appearanceOutput/keyboard-prefix-4.json"))
    & $appearancePython -B -c "import json;from pathlib import Path;from bench.quality_evaluate import prefix_equal;from bench.quality_assets import save_result;p=Path('$appearanceOutput');a=json.loads((p/'keyboard-prefix-2.json').read_text());b=json.loads((p/'keyboard-prefix-4.json').read_text());r=dict(same_provenance=a['provenance']==b['provenance'],prefix_identical=prefix_equal(a,b),scope='Two versus four original keyboard frames; full prefix gate still required');save_result(p/'repeatability.json',r);print(r);assert r['same_provenance']"
    if ($LASTEXITCODE -ne 0) { throw 'Prefix provenance check failed' }
    & $appearancePython -B -m bench.quality_trace_compare --short "$appearanceOutput/repeat-short-trace.jsonl" --long "$appearanceOutput/repeat-long-trace.jsonl" --output "$appearanceOutput/repeat-trace-comparison.json"
    if ($LASTEXITCODE -ne 0) { throw 'Trace comparison failed' }
    $appearancePose = @('pose')+$appearanceCommon+@('--masks','.cache/model-quality/results/ranch/segmentation','--mode','complete')
    Invoke-AppearanceStage 'prefix-control' ($appearancePose+@('--limit','30','--output',"$appearanceOutput/prefix-30/without-appearance.json"))
    Invoke-AppearanceStage 'prefix-appearance' ($appearancePose+@('--limit','30','--appearance-check','--output',"$appearanceOutput/prefix-30/complete.json"))
    & $appearancePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Incremental appearance viewer export failed' }
    Invoke-AppearanceStage 'full-control' ($appearancePose+@('--output',"$appearanceOutput/without-appearance.json"))
    Invoke-AppearanceStage 'full-appearance' ($appearancePose+@('--appearance-check','--output',"$appearanceOutput/complete.json"))
    & $appearancePython -B -m bench.quality_appearance_benchmark
    if ($LASTEXITCODE -ne 0) { throw 'Appearance experiment report failed' }
    & $appearancePython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Appearance viewer export failed' }
} finally { Pop-Location }
