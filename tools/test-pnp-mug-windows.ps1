[CmdletBinding()]
param(
    [string]$OutputRoot,
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [switch]$TestMode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-StructuralEqual($left, $right) {
    if ($null -eq $left -or $null -eq $right) { return ($null -eq $left -and $null -eq $right) }
    $leftMap = $left -is [System.Collections.IDictionary] -or $left -is [System.Management.Automation.PSCustomObject]
    $rightMap = $right -is [System.Collections.IDictionary] -or $right -is [System.Management.Automation.PSCustomObject]
    if ($leftMap -or $rightMap) {
        if (-not ($leftMap -and $rightMap)) { return $false }
        if ($left -is [System.Collections.IDictionary]) { $leftKeys = @($left.Keys | ForEach-Object { [string]$_ }) }
        else { $leftKeys = @($left.PSObject.Properties | ForEach-Object { $_.Name }) }
        if ($right -is [System.Collections.IDictionary]) { $rightKeys = @($right.Keys | ForEach-Object { [string]$_ }) }
        else { $rightKeys = @($right.PSObject.Properties | ForEach-Object { $_.Name }) }
        if ($leftKeys.Count -ne $rightKeys.Count) { return $false }
        foreach ($key in $leftKeys) {
            if ($right -is [System.Collections.IDictionary]) {
                if (-not $right.Contains($key)) { return $false }
                $rightValue = $right[$key]
            } else {
                $rightProperty = $right.PSObject.Properties[$key]
                if ($null -eq $rightProperty) { return $false }
                $rightValue = $rightProperty.Value
            }
            if ($left -is [System.Collections.IDictionary]) { $leftValue = $left[$key] }
            else { $leftValue = $left.PSObject.Properties[$key].Value }
            if (-not (Test-StructuralEqual $leftValue $rightValue)) { return $false }
        }
        return $true
    }
    $leftList = $left -is [System.Collections.IList]
    $rightList = $right -is [System.Collections.IList]
    if ($leftList -or $rightList) {
        if (-not ($leftList -and $rightList) -or $left.Count -ne $right.Count) { return $false }
        for ($index = 0; $index -lt $left.Count; $index++) {
            if (-not (Test-StructuralEqual $left[$index] $right[$index])) { return $false }
        }
        return $true
    }
    $leftJson = ConvertTo-Json -InputObject $left -Compress -Depth 100
    $rightJson = ConvertTo-Json -InputObject $right -Compress -Depth 100
    return [string]::Equals($leftJson, $rightJson, [StringComparison]::Ordinal)
}

function Test-HashMapAgainstRoot($hashMap, [string]$root) {
    $changed = [System.Collections.Generic.List[string]]::new()
    if ($hashMap -is [System.Collections.IDictionary]) {
        $entries = @($hashMap.GetEnumerator())
    } elseif ($hashMap -is [System.Management.Automation.PSCustomObject]) {
        $entries = @($hashMap.PSObject.Properties)
    } else {
        return [pscustomobject]@{ Passed = $false; Changed = @('<invalid hash map>') }
    }
    foreach ($entry in $entries) {
        if ($entry -is [System.Collections.DictionaryEntry]) {
            $relative = [string]$entry.Key
            $expected = [string]$entry.Value
        } else {
            $relative = [string]$entry.Name
            $expected = [string]$entry.Value
        }
        $path = Join-Path $root $relative.Replace('/', [IO.Path]::DirectorySeparatorChar)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $changed.Add($relative)
            continue
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected.ToLowerInvariant()) { $changed.Add($relative) }
    }
    return [pscustomobject]@{ Passed = ($changed.Count -eq 0); Changed = @($changed.ToArray()) }
}

function Test-TraceRecord($traceRecord) {
    if ($null -eq $traceRecord) { return $false }
    $stageProperty = $traceRecord.PSObject.Properties['stage']
    $arraysProperty = $traceRecord.PSObject.Properties['arrays']
    if ($null -eq $stageProperty -or $null -eq $arraysProperty -or
        [string]::IsNullOrWhiteSpace([string]$stageProperty.Value) -or $null -eq $arraysProperty.Value) {
        return $false
    }
    return (@($arraysProperty.Value.PSObject.Properties).Count -gt 0)
}

function Invoke-PureCpuFixtures {
    $first = '{"source":{"a":"hash-a","b":"hash-b"},"ids":[827,828],"provenance":{"bank":"bank-a","input":"input-a"}}' | ConvertFrom-Json
    $reordered = '{"provenance":{"input":"input-a","bank":"bank-a"},"ids":[827,828],"source":{"b":"hash-b","a":"hash-a"}}' | ConvertFrom-Json
    $changedBank = '{"provenance":{"input":"input-a","bank":"bank-b"},"ids":[827,828],"source":{"b":"hash-b","a":"hash-a"}}' | ConvertFrom-Json
    $changedSource = '{"provenance":{"input":"input-a","bank":"bank-a"},"ids":[827,828],"source":{"b":"hash-b","a":"edited"}}' | ConvertFrom-Json
    $changedOrder = '{"provenance":{"input":"input-a","bank":"bank-a"},"ids":[828,827],"source":{"b":"hash-b","a":"hash-a"}}' | ConvertFrom-Json
    if (-not (Test-StructuralEqual $first $reordered)) { throw 'Structural equality rejected equivalent nested evidence objects.' }
    if (Test-StructuralEqual $first $changedBank) { throw 'Structural equality accepted a changed bank hash.' }
    if (Test-StructuralEqual $first $changedSource) { throw 'Structural equality accepted a changed source hash.' }
    if (Test-StructuralEqual $first $changedOrder) { throw 'Structural equality ignored ordered ID differences.' }
    $validTrace = '{"stage":"render","arrays":{"camera_pose":{"shape":[4,4],"dtype":"float64","sha256":"pose-hash","mean":1.0},"color":{"shape":[2,2,3],"dtype":"uint8","sha256":"color-hash","mean":0.0}}}' | ConvertFrom-Json
    $emptyTrace = '{"stage":"render","arrays":{}}' | ConvertFrom-Json
    if (-not (Test-TraceRecord $validTrace)) { throw 'Trace-record validator rejected nonempty render arrays.' }
    if (Test-TraceRecord $emptyTrace) { throw 'Trace-record validator accepted empty render arrays.' }

    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ('visualizeit-pnp-cpu-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tempRoot -ErrorAction Stop | Out-Null
    try {
        $fixture = Join-Path $tempRoot 'checkpoint.bin'
        [IO.File]::WriteAllBytes($fixture, [byte[]](1, 2, 3))
        $expected = (Get-FileHash -LiteralPath $fixture -Algorithm SHA256).Hash.ToLowerInvariant()
        $frozenMap = [ordered]@{ 'checkpoint.bin' = $expected }
        if (-not (Test-HashMapAgainstRoot $frozenMap $tempRoot).Passed) {
            throw 'Frozen-file hash fixture did not accept unchanged bytes.'
        }
        [IO.File]::WriteAllBytes($fixture, [byte[]](1, 2, 4))
        $changed = Test-HashMapAgainstRoot $frozenMap $tempRoot
        if ($changed.Passed -or $changed.Changed -notcontains 'checkpoint.bin') {
            throw 'Frozen-file hash fixture did not reject changed bytes.'
        }
    } finally {
        $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        $resolvedFixtureRoot = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $tempRoot).Path)
        if ($resolvedFixtureRoot.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetFileName($resolvedFixtureRoot).StartsWith('visualizeit-pnp-cpu-', [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedFixtureRoot -Recurse -Force
        }
    }
    Write-Output 'PowerShell CPU structural and frozen-file fixtures passed.'
}

if ($TestMode) {
    Invoke-PureCpuFixtures
    return
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { throw 'OutputRoot is required for a real experiment.' }

$projectRoot = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path)
$directorySeparator = [IO.Path]::DirectorySeparatorChar
$callerPath = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath (Get-Location).Path).Path)
$projectPrefix = $projectRoot.TrimEnd('\', '/') + $directorySeparator
if (-not $callerPath.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase) -and
    -not [string]::Equals($callerPath, $projectRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Run this harness from inside the VisualizeIT workspace.'
}

$requestedOutputRoot = if ([IO.Path]::IsPathRooted($OutputRoot)) {
    [IO.Path]::GetFullPath($OutputRoot)
} else {
    [IO.Path]::GetFullPath((Join-Path $callerPath $OutputRoot))
}
$requestedParent = Split-Path -Parent $requestedOutputRoot
$requestedLeaf = Split-Path -Leaf $requestedOutputRoot
if (-not (Test-Path -LiteralPath $requestedParent -PathType Container)) {
    throw 'OutputRoot parent must already exist inside the VisualizeIT workspace.'
}
$resolvedParent = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $requestedParent).Path)
$resolvedOutputRoot = [IO.Path]::GetFullPath((Join-Path $resolvedParent $requestedLeaf))
if ([string]::Equals($resolvedOutputRoot, $projectRoot, [StringComparison]::OrdinalIgnoreCase) -or
    -not $resolvedOutputRoot.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputRoot must be a new directory inside the VisualizeIT workspace.'
}
if (Test-Path -LiteralPath $resolvedOutputRoot) {
    throw 'OutputRoot already exists; preserve it and choose a fresh directory.'
}
New-Item -ItemType Directory -Path $resolvedOutputRoot -ErrorAction Stop | Out-Null

$python = Join-Path $projectRoot '.cache/quality-windows/Scripts/python.exe'
$bundle = Join-Path $projectRoot '.cache/model-quality/inputs/mug'
$masks = Join-Path $projectRoot '.cache/model-quality/results/mug/segmentation'
$smoke = Join-Path $projectRoot ('.cache/model-quality/smoke/mug-' + $Device + '-unlit-no-msaa.json')
$evidencePath = Join-Path $resolvedOutputRoot 'stage-evidence.json'
$scriptHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$stageSpecs = @(
    [ordered]@{ id = 'control-30'; branch = 'control'; length = 30; pnp = $true },
    [ordered]@{ id = 'control-120'; branch = 'control'; length = 120; pnp = $true },
    [ordered]@{ id = 'control-240'; branch = 'control'; length = 240; pnp = $true },
    [ordered]@{ id = 'candidate-30'; branch = 'candidate'; length = 30; pnp = $false },
    [ordered]@{ id = 'candidate-120'; branch = 'candidate'; length = 120; pnp = $false },
    [ordered]@{ id = 'candidate-240'; branch = 'candidate'; length = 240; pnp = $false }
)
$expectedStageIds = @($stageSpecs | ForEach-Object { $_.id })
$evidence = [ordered]@{
    schema_version = 1
    status = 'preparing'
    overall_status = 'running'
    created_utc = [DateTime]::UtcNow.ToString('o')
    device = $Device
    output_root = $resolvedOutputRoot
    harness_sha256 = $scriptHash
    attempted_stage_ids = [System.Collections.Generic.List[string]]::new()
    stages = [System.Collections.Generic.List[object]]::new()
    evaluation = [ordered]@{ status = 'pending'; output = 'report.json'; log = 'evaluation.log' }
    failure = $null
}
foreach ($spec in $stageSpecs) {
    $requestedIds = @(827..(826 + $spec.length))
    $evidence.stages.Add([ordered]@{
        stage_id = $spec.id
        status = 'pending'
        exit_code = $null
        terminal_exit_recorded = $false
        output = ($spec.id + '.json')
        output_sha256 = $null
        trace = ($spec.id + '.trace.jsonl')
        trace_sha256 = $null
        requested_frame_ids = $requestedIds
        expected_complete = ($spec.length -eq 240)
        pnp_use_extrinsic_guess = $spec.pnp
    })
}

function Save-Evidence {
    $temporary = $evidencePath + '.tmp'
    $json = $evidence | ConvertTo-Json -Depth 100
    [IO.File]::WriteAllText($temporary, $json + "`n", [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $evidencePath -Force | Out-Null
}

function Assert-SourceIntact($snapshot, [string]$stageId) {
    $changed = [System.Collections.Generic.List[string]]::new()
    $sourceCheck = Test-HashMapAgainstRoot $snapshot.source_files $projectRoot
    $frozenCheck = Test-HashMapAgainstRoot $snapshot.frozen_files $projectRoot
    foreach ($relative in $sourceCheck.Changed) { $changed.Add("source:$relative") }
    foreach ($relative in $frozenCheck.Changed) { $changed.Add("input:$relative") }
    $currentHarnessHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($currentHarnessHash -ne $scriptHash) { $changed.Add('tools/test-pnp-mug-windows.ps1') }
    if ($changed.Count -gt 0) {
        throw "Frozen source or input changed before or during ${stageId}: $($changed -join ', ')"
    }
}

function Assert-NoOtherPoseProcess([string]$stageId) {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match '-m bench\.quality_runner'
    }
    if ($active) { throw "Another quality_runner process is active before $stageId." }
}

function Assert-StageOutput($record, $spec, $snapshot) {
    $outputPath = Join-Path $resolvedOutputRoot $record.output
    if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
        throw "Stage $($spec.id) exited without its output file."
    }
    $result = Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json
    $expectedIds = @(827..(826 + $spec.length))
    if ($result.object -ne 'mug' -or $result.mode -ne 'complete') {
        throw "Stage $($spec.id) did not produce a mug complete-mode result."
    }
    if ($result.frame_ids.Count -ne $expectedIds.Count) { throw "Stage $($spec.id) has the wrong frame count." }
    for ($index = 0; $index -lt $expectedIds.Count; $index++) {
        if ([int]$result.frame_ids[$index] -ne $expectedIds[$index]) {
            throw "Stage $($spec.id) has incorrect frame order at offset $index."
        }
    }
    if ($result.complete -isnot [bool] -or $result.complete -ne ($spec.length -eq 240)) {
        throw "Stage $($spec.id) has the wrong complete flag."
    }
    if ($result.PSObject.Properties.Name -contains 'status' -and $result.status -ne 'complete') {
        throw "Stage $($spec.id) is not terminal complete."
    }
    if ($result.automatic -ne $true -or $result.diagnostic_control -ne $false -or
        $result.independent_accuracy_scored -ne $false -or $null -ne $result.stress_test) {
        throw "Stage $($spec.id) has disallowed result provenance or scoring flags."
    }
    if ($result.tracking_settings.pnp_use_extrinsic_guess -isnot [bool] -or
        $result.tracking_settings.pnp_use_extrinsic_guess -ne $spec.pnp -or
        $result.tracking_settings.model_memory -ne $false -or
        $result.tracking_settings.unlit_templates -ne $true -or
        $result.tracking_settings.appearance_check -ne $false -or
        $result.tracking_settings.disable_multisampling -ne $true) {
        throw "Stage $($spec.id) changed a required branch or fixed setting."
    }
    if ($result.initialization.frameId -ne 826 -or $null -ne $result.initialization.cameraFromObject) {
        throw "Stage $($spec.id) did not use setup-only initialization."
    }
    $tracePath = Join-Path $resolvedOutputRoot $record.trace
    if (-not (Test-Path -LiteralPath $tracePath -PathType Leaf)) {
        throw "Stage $($spec.id) exited without its required trace file."
    }
    $traceBytes = [IO.File]::ReadAllBytes($tracePath)
    if ($traceBytes.Length -lt 2 -or $traceBytes[$traceBytes.Length - 1] -ne 10) {
        throw "Stage $($spec.id) trace is empty or truncated before its final newline."
    }
    $traceText = [Text.UTF8Encoding]::new($false, $true).GetString($traceBytes)
    $traceLines = @($traceText -split "`n")
    if ($traceLines[-1] -eq '') { $traceLines = @($traceLines[0..($traceLines.Count - 2)]) }
    if ($traceLines.Count -lt 1 -or @($traceLines | Where-Object { [string]::IsNullOrWhiteSpace($_) }).Count -gt 0) {
        throw "Stage $($spec.id) trace is empty or contains a blank record."
    }
    foreach ($line in $traceLines) {
        try { $traceRecord = $line | ConvertFrom-Json } catch { throw "Stage $($spec.id) trace contains invalid JSON." }
        if (-not (Test-TraceRecord $traceRecord)) {
            throw "Stage $($spec.id) trace contains an empty record."
        }
    }
    $record.result_frame_ids = @($result.frame_ids)
    $record.result_complete = $result.complete
    $record.runtime = $result.runtime
    $record.tracking_settings = $result.tracking_settings
    $record.inference_provenance = $result.provenance
    $record.output_sha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $record.trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $record.trace_record_count = $traceLines.Count
    $record.source_files = $snapshot.source_files
    $record.frozen_files = $snapshot.frozen_files
}

Save-Evidence
$allStagesComplete = $false
try {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Pinned Python executable is missing: $python" }
    if (-not (Test-Path -LiteralPath $bundle -PathType Container) -or
        -not (Test-Path -LiteralPath $masks -PathType Container) -or
        -not (Test-Path -LiteralPath $smoke -PathType Leaf)) {
        throw 'The frozen mug bundle, automatic mask cache, or existing renderer smoke is missing.'
    }
    Assert-NoOtherPoseProcess 'snapshot'
    $snapshotLog = Join-Path $resolvedOutputRoot 'snapshot.log'
    & $python -B -m bench.quality_pnp_mug_report snapshot --bundle $bundle --masks $masks --experiment-root $resolvedOutputRoot --smoke $smoke *> $snapshotLog
    $snapshotExit = $LASTEXITCODE
    if ($snapshotExit -ne 0) { throw "Snapshot preparation exited with code $snapshotExit. See snapshot.log." }
    $snapshot = Get-Content -LiteralPath (Join-Path $resolvedOutputRoot 'experiment.json') -Raw | ConvertFrom-Json
    if ($snapshot.expected_frame_ids.Count -ne 240 -or $snapshot.setup_frame_id -ne 826) {
        throw 'The snapshot does not record the canonical setup and 240 scored frames.'
    }
    $evidence.status = 'running'
    $evidence.experiment_sha256 = (Get-FileHash -LiteralPath (Join-Path $resolvedOutputRoot 'experiment.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    $evidence.source_files = $snapshot.source_files
    $evidence.frozen_files = $snapshot.frozen_files
    $evidence.inference_provenance = $snapshot.inference_provenance
    Save-Evidence

    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    $env:PYTHONHASHSEED = '0'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    foreach ($spec in $stageSpecs) {
        Assert-NoOtherPoseProcess $spec.id
        Assert-SourceIntact $snapshot $spec.id
        $record = $evidence.stages | Where-Object { $_.stage_id -eq $spec.id } | Select-Object -First 1
        $evidence.attempted_stage_ids.Add($spec.id)
        $record.status = 'running'
        $record.started_utc = [DateTime]::UtcNow.ToString('o')
        $record.source_files = $snapshot.source_files
        $record.frozen_files = $snapshot.frozen_files
        $record.inference_provenance = $snapshot.inference_provenance
        $record.arguments = @('pose', '--bundle', $bundle, '--masks', $masks, '--output', (Join-Path $resolvedOutputRoot $record.output),
            '--device', $Device, '--mode', 'complete', '--unlit-templates', '--disable-multisampling')
        if ($spec.length -lt 240) { $record.arguments += @('--limit', [string]$spec.length) }
        if (-not $spec.pnp) { $record.arguments += '--pnp-no-extrinsic-guess' }
        $tracePath = Join-Path $resolvedOutputRoot $record.trace
        if (Test-Path -LiteralPath $tracePath) { throw "Stage trace already exists: $tracePath" }
        $env:VISUALIZEIT_QUALITY_TRACE = $tracePath
        $logPath = Join-Path $resolvedOutputRoot ($spec.id + '.log')
        $arguments = [string[]]$record.arguments
        Save-Evidence
        try {
            & $python -B -m bench.quality_runner @arguments *> $logPath
            $record.exit_code = $LASTEXITCODE
            $record.terminal_exit_recorded = $true
            $record.finished_utc = [DateTime]::UtcNow.ToString('o')
            if ($record.exit_code -ne 0) { throw "Process exited with code $($record.exit_code). See $($spec.id).log." }
            Assert-StageOutput $record $spec $snapshot
            if (-not (Test-StructuralEqual $record.source_files $snapshot.source_files) -or
                -not (Test-StructuralEqual $record.frozen_files $snapshot.frozen_files) -or
                -not (Test-StructuralEqual $record.inference_provenance $snapshot.inference_provenance)) {
                throw "Stage $($spec.id) did not retain frozen provenance."
            }
            Assert-SourceIntact $snapshot $spec.id
            $record.status = 'succeeded'
            Save-Evidence
        } catch {
            $record.status = 'failed'
            $record.error = $_.Exception.Message
            $record.finished_utc = [DateTime]::UtcNow.ToString('o')
            $evidence.status = 'failed'
            $evidence.overall_status = 'failed'
            $evidence.failure = [ordered]@{ stage_id = $spec.id; message = $_.Exception.Message; utc = [DateTime]::UtcNow.ToString('o') }
            Save-Evidence
            throw
        }
    }
    $allStagesComplete = $true
    $evidence.status = 'complete'
    $evidence.completed_utc = [DateTime]::UtcNow.ToString('o')
    $evidence.evaluation.status = 'running'
    Save-Evidence

    Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue
    $evaluationLog = Join-Path $resolvedOutputRoot 'evaluation.log'
    $reportPath = Join-Path $resolvedOutputRoot 'report.json'
    $evidence.evaluation.started_utc = [DateTime]::UtcNow.ToString('o')
    Save-Evidence
    & $python -B -m bench.quality_pnp_mug_report evaluate --experiment-root $resolvedOutputRoot --output $reportPath *> $evaluationLog
    $evaluationExit = $LASTEXITCODE
    $evidence.evaluation.exit_code = $evaluationExit
    $evidence.evaluation.terminal_exit_recorded = $true
    $evidence.evaluation.finished_utc = [DateTime]::UtcNow.ToString('o')
    if ($evaluationExit -ne 0) {
        $evidence.evaluation.status = 'failed'
        $evidence.evaluation.error = "Evaluator exited with code $evaluationExit. See evaluation.log."
        $evidence.overall_status = 'report_failed'
        Save-Evidence
        throw $evidence.evaluation.error
    }
    $evidence.evaluation.status = 'succeeded'
    $evidence.evaluation.output_sha256 = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $evidence.overall_status = 'report_written'
    Save-Evidence
    Write-Output "Completed six terminal pose stages. Report: $reportPath"
} catch {
    if (-not $allStagesComplete) {
        $evidence.status = 'failed'
        $evidence.overall_status = 'failed'
        if ($null -eq $evidence.failure) {
            $evidence.failure = [ordered]@{ stage_id = $null; message = $_.Exception.Message; utc = [DateTime]::UtcNow.ToString('o') }
        }
    }
    if ($allStagesComplete -and $evidence.evaluation.status -eq 'running') {
        $evidence.evaluation.status = 'failed'
        $evidence.evaluation.error = $_.Exception.Message
        $evidence.evaluation.finished_utc = [DateTime]::UtcNow.ToString('o')
        $evidence.overall_status = 'report_failed'
    }
    Save-Evidence
    throw
} finally {
    Remove-Item Env:VISUALIZEIT_QUALITY_TRACE -ErrorAction SilentlyContinue
}
