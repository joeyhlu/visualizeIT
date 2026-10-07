param([int]$WaitForProcess = 0)
$ErrorActionPreference = 'Stop'
$memoryRoot = Split-Path -Parent $PSScriptRoot
$memoryPython = Join-Path $memoryRoot '.cache/quality-windows/Scripts/python.exe'
Push-Location -LiteralPath $memoryRoot
try {
    if ($WaitForProcess -gt 0) {
        $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$WaitForProcess"
        if ($existing -and $existing.CommandLine -notmatch 'run-quality-windows\.ps1') {
            throw 'Refusing to wait on a process unrelated to the existing benchmark.'
        }
        if ($existing) {
            Write-Output "Waiting for existing benchmark process $WaitForProcess; no concurrent neural GPU stages."
            Wait-Process -Id $WaitForProcess -ErrorAction SilentlyContinue
        }
    }
    $memoryOutput = '.cache/model-quality/results/keyboard/model-memory'
    New-Item -ItemType Directory -Force $memoryOutput | Out-Null
    function Invoke-MemoryStage([string]$Name, [string[]]$Arguments) {
        Write-Output "Starting $Name"
        & $memoryPython -B -m bench.quality_runner @Arguments *> "$memoryOutput/$Name.log"
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath "$memoryOutput/$Name.log" -Tail 30
            throw "Model-memory experiment failed at $Name; preserve the failed stage."
        }
    }
    $memoryCommon = @('--bundle','.cache/model-quality/inputs/keyboard','--device','cuda')
    Invoke-MemoryStage 'smoke' (@('smoke')+$memoryCommon+@('--output','.cache/model-quality/smoke/keyboard-cuda.json'))
    Invoke-MemoryStage 'sam-prefix' (@('segment')+$memoryCommon+@('--output',"$memoryOutput/prefix-30/segmentation",'--limit','30'))
    Invoke-MemoryStage 'sam-full' (@('segment')+$memoryCommon+@('--output',"$memoryOutput/segmentation"))
    Invoke-MemoryStage 'pose-prefix' (@('pose')+$memoryCommon+@('--masks',"$memoryOutput/prefix-30/segmentation",'--output',"$memoryOutput/prefix-30/complete.json",'--mode','complete','--limit','30','--model-memory'))
    & $memoryPython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Incremental model-memory viewer export failed' }
    Invoke-MemoryStage 'pose-without-memory' (@('pose')+$memoryCommon+@('--masks',"$memoryOutput/segmentation",'--output',"$memoryOutput/without-memory.json",'--mode','complete'))
    Invoke-MemoryStage 'pose-with-memory' (@('pose')+$memoryCommon+@('--masks',"$memoryOutput/segmentation",'--output',"$memoryOutput/complete.json",'--mode','complete','--model-memory'))
    & $memoryPython -B -m bench.quality_memory_report
    if ($LASTEXITCODE -ne 0) { throw 'Model-memory report failed' }
    & $memoryPython -B -m bench.quality_player
    if ($LASTEXITCODE -ne 0) { throw 'Model-memory viewer export failed' }
} finally { Pop-Location }
