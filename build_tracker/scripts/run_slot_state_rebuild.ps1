# One-shot offline rebuild/export supervisor. Run from a durable scheduled task.
param([string]$TaskName = 'BuildTracker 16.18 Role Slots Rebuild')

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$chainRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $chainRoot
$chainPython = Join-Path $chainRoot '.venv\Scripts\python.exe'
$chainSource = Join-Path $chainRoot 'data\corpora\16.18_initial'
$chainCorpus = Join-Path $chainRoot 'data\corpora\16.18_role_slots_v1'
$chainEvidence = Join-Path $chainRoot 'data\acceptance\16.18_role_slots_v1'
$chainCoverage = Join-Path $chainEvidence 'coverage'
$chainGeneration = Join-Path $chainRoot 'data\dataset_generations\16.18-initial-role-slots-v1'
$chainPrior = Join-Path $chainRoot 'data\dataset_generations\16.18-initial-smoke-v2'
$chainLog = Join-Path $chainEvidence 'chain.log'
$chainStatusPath = Join-Path $chainEvidence 'chain_status.json'
$chainInitialized = $false
$chainExit = 1
$chainStage = 'preflight'
$chainCodeBefore = [ordered]@{}
$chainProtectedBefore = [ordered]@{}
$chainInputsBefore = [ordered]@{}
$chainCompleted = New-Object 'System.Collections.Generic.List[string]'
$chainStarted = [DateTime]::UtcNow.ToString('o')

function Get-ChainHashes([string[]]$Paths, [switch]$AllowMissing) {
    $hashes = [ordered]@{}
    foreach ($relative in $Paths) {
        $path = Join-Path $chainRoot $relative
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $hashes[$relative] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        } elseif ($AllowMissing) {
            $hashes[$relative] = $null
        } else {
            throw "Required input is missing: $relative"
        }
    }
    return ,$hashes
}

function Assert-ChainHashes($Expected, [string]$Label, [switch]$AllowMissing) {
    $actual = Get-ChainHashes -Paths @($Expected.Keys) -AllowMissing:$AllowMissing
    foreach ($relative in $Expected.Keys) {
        if ($actual[$relative] -ne $Expected[$relative]) {
            throw "$Label changed during rebuild: $relative"
        }
    }
}

function Assert-ChainUnchanged {
    Assert-ChainHashes -Expected $chainCodeBefore -Label 'Pinned code'
    Assert-ChainHashes -Expected $chainInputsBefore -Label 'Pinned source/static/split metadata'
    Assert-ChainHashes -Expected $chainProtectedBefore -Label 'Protected serving/hold artifact' -AllowMissing
}

function Append-ChainLog([string]$Message) {
    # A viewer/tailing process can briefly hold a conflicting Windows handle.
    for ($attempt = 0; $attempt -lt 8; $attempt++) {
        try {
            Add-Content -LiteralPath $chainLog -Value $Message -Encoding UTF8 -ErrorAction Stop
            return
        } catch {
            if ($attempt -eq 7) { throw }
            Start-Sleep -Milliseconds ([int](50 * [Math]::Pow(2, $attempt)))
        }
    }
}

function Write-ChainLog([string]$Message) {
    Append-ChainLog ('[' + [DateTime]::UtcNow.ToString('o') + '] ' + $Message)
}

function Write-ChainStatus([string]$State, [string]$ErrorMessage = '') {
    $protectedAfter = Get-ChainHashes -Paths @($chainProtectedBefore.Keys) -AllowMissing
    $status = [ordered]@{
        schema = 1
        state = $State
        stage = $chainStage
        started_utc = $chainStarted
        updated_utc = [DateTime]::UtcNow.ToString('o')
        task_name = $TaskName
        pid = $PID
        source = $chainSource
        corpus = $chainCorpus
        coverage = $chainCoverage
        generation = $chainGeneration
        prior_generation = $chainPrior
        completed_stages = @($chainCompleted.ToArray())
        code_sha256 = $chainCodeBefore
        input_metadata_sha256 = $chainInputsBefore
        protected_before = $chainProtectedBefore
        protected_after = $protectedAfter
        final_test_evaluation = $false
        training_run = $false
        production_hold_cleared = $false
        promotion_allowed = $false
        error = $ErrorMessage
    }
    $temporary = Join-Path $chainEvidence ('.chain_status.' + $PID + '.' + [Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        $encoding = New-Object System.Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($temporary, (($status | ConvertTo-Json -Depth 12) + [Environment]::NewLine), $encoding)
        for ($attempt = 0; $attempt -lt 8; $attempt++) {
            try {
                if (Test-Path -LiteralPath $chainStatusPath) {
                    [IO.File]::Replace($temporary, $chainStatusPath, [NullString]::Value)
                } else {
                    [IO.File]::Move($temporary, $chainStatusPath)
                }
                break
            } catch {
                if ($attempt -eq 7) { throw }
                Start-Sleep -Milliseconds ([int](50 * [Math]::Pow(2, $attempt)))
            }
        }
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -ErrorAction SilentlyContinue }
    }
}

function Invoke-ChainStage([string]$Name, [string[]]$Arguments) {
    $script:chainStage = $Name
    Assert-ChainUnchanged
    Write-ChainStatus -State 'running'
    Write-ChainLog ('Starting ' + $Name + ': ' + ($Arguments -join ' '))
    $savedPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 wraps native stderr in ErrorRecord objects.
        $ErrorActionPreference = 'Continue'
        & $chainPython @Arguments 2>&1 | ForEach-Object {
            Append-ChainLog ([string]$_)
        }
        $stageExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($stageExit -ne 0) { throw "$Name failed with exit code $stageExit; see chain.log" }
    Assert-ChainUnchanged
    $chainCompleted.Add($Name)
    Write-ChainLog ('Completed ' + $Name)
    Write-ChainStatus -State 'running'
}

try {
    # Never attach a second run to an existing evidence directory or output.
    if (Test-Path -LiteralPath $chainEvidence) { throw "Evidence output already exists: $chainEvidence" }
    New-Item -ItemType Directory -Path $chainEvidence -ErrorAction Stop | Out-Null
    $chainInitialized = $true
    Write-ChainLog 'Starting offline role-slot rebuild, audit, qualification and export. No collection or model training.'
    foreach ($output in @($chainCorpus, $chainGeneration)) {
        if (Test-Path -LiteralPath $output) { throw "Immutable output already exists: $output" }
    }
    if (-not (Test-Path -LiteralPath $chainPython -PathType Leaf)) { throw 'The repository venv interpreter is missing' }
    $chainScripts = @(
        'scripts/run_slot_state_rebuild.ps1', 'scripts/rebuild_frozen_corpus.py',
        'scripts/audit_patch_corpus.py', 'scripts/qualify_patch_corpus.py',
        'scripts/report_corpus_coverage.py', 'scripts/corpus_rows.py',
        'scripts/baseline.py', 'scripts/collect_patch_sample.py',
        'scripts/publish_patch_dataset.py', 'scripts/verify_slot_state_export.py'
    )
    $chainAppFiles = @(Get-ChildItem -LiteralPath (Join-Path $chainRoot 'app') -Filter '*.py' -File | Sort-Object Name | ForEach-Object { 'app/' + $_.Name })
    $chainCodeBefore = Get-ChainHashes -Paths @($chainScripts + $chainAppFiles)
    $chainProtectedBefore = Get-ChainHashes -Paths @(
        'data/ml/prefix_model.pt', 'data/ml/deployment_manifest.json', 'data/ml/pipeline_hold.json'
    ) -AllowMissing
    $sourceManifest = Get-Content -LiteralPath (Join-Path $chainSource 'corpus_manifest.json') -Raw | ConvertFrom-Json
    $staticVersion = [string]$sourceManifest.ddragon_version
    if ($staticVersion -notmatch '^16\.18\.[0-9]+$') { throw 'Unexpected frozen static-data version' }
    $chainInputPaths = @(
        'data/corpora/16.18_initial/corpus_manifest.json',
        'data/corpora/16.18_initial/source_inventory.jsonl',
        'data/dataset_generations/16.18-initial-smoke-v2/generation_manifest.json',
        'data/dataset_generations/16.18-initial-smoke-v2/split_manifest.json',
        'data/dataset_generations/16.18-initial-smoke-v2/split_assignment.json',
        'data/dataset_generations/16.18-initial-smoke-v2/split_plan.json'
    )
    foreach ($kind in @('item', 'champion')) {
        $cacheRelative = 'data/cache/' + $kind + '-' + $staticVersion + '.json'
        $sourceRelative = 'data/corpora/16.18_initial/static/' + $kind + '-' + $staticVersion + '.json'
        $staticHashes = Get-ChainHashes -Paths @($cacheRelative, $sourceRelative)
        if ($staticHashes[$cacheRelative] -ne $staticHashes[$sourceRelative]) { throw "Cached $kind data differs from frozen source" }
        $chainInputPaths += @($cacheRelative, $sourceRelative)
    }
    $chainInputsBefore = Get-ChainHashes -Paths $chainInputPaths
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    # All required static files are pinned above. Any unexpected HTTP fallback
    # fails locally rather than fetching replacement inputs during this chain.
    $env:HTTP_PROXY = 'http://127.0.0.1:9'
    $env:HTTPS_PROXY = 'http://127.0.0.1:9'
    $env:ALL_PROXY = 'http://127.0.0.1:9'
    $env:NO_PROXY = ''
    Assert-ChainUnchanged
    Write-ChainStatus -State 'running'
    Invoke-ChainStage -Name 'rebuild' -Arguments @('scripts/rebuild_frozen_corpus.py', '--source', $chainSource, '--out', $chainCorpus)
    Invoke-ChainStage -Name 'semantic_audit' -Arguments @('scripts/audit_patch_corpus.py', '--corpus', $chainCorpus, '--out', (Join-Path $chainEvidence 'semantic_audit.json'))
    Invoke-ChainStage -Name 'qualification' -Arguments @('scripts/qualify_patch_corpus.py', '--corpus', $chainCorpus, '--audit', (Join-Path $chainEvidence 'semantic_audit.json'), '--out', (Join-Path $chainEvidence 'qualification.json'))
    Invoke-ChainStage -Name 'coverage' -Arguments @('scripts/report_corpus_coverage.py', '--qualification', (Join-Path $chainEvidence 'qualification.json'), '--out', $chainCoverage, '--benchmark-id', '16.18-initial-role-slots-v1', '--eval-version', '3')
    Invoke-ChainStage -Name 'split_preservation' -Arguments @('scripts/verify_slot_state_export.py', '--coverage', $chainCoverage, '--prior-generation', $chainPrior, '--out', (Join-Path $chainEvidence 'split_preservation.json'))
    Invoke-ChainStage -Name 'export' -Arguments @('scripts/publish_patch_dataset.py', '--coverage', $chainCoverage, '--name', '16.18-initial-role-slots-v1')
    Invoke-ChainStage -Name 'export_verification' -Arguments @('scripts/verify_slot_state_export.py', '--generation', $chainGeneration, '--prior-generation', $chainPrior, '--out', (Join-Path $chainEvidence 'export_verification.json'))
    $chainStage = 'complete'
    Assert-ChainUnchanged
    Write-ChainLog 'All seven stages completed; protected serving/hold artifacts and pinned inputs are unchanged.'
    Write-ChainStatus -State 'complete'
    $chainExit = 0
} catch {
    $chainFailure = $_.Exception.Message
    if ($chainInitialized) {
        try { Assert-ChainUnchanged } catch { $chainFailure += '; ' + $_.Exception.Message }
        try { Write-ChainLog ('CHAIN FAILED: ' + $chainFailure) } catch { [Console]::Error.WriteLine($_.Exception.Message) }
        try { Write-ChainStatus -State 'failed' -ErrorMessage $chainFailure } catch { [Console]::Error.WriteLine($_.Exception.Message) }
    }
    [Console]::Error.WriteLine($chainFailure)
} finally {
    # This supervisor owns exactly its named one-shot task, never other jobs.
    if ($TaskName) {
        try {
            $ownTask = Get-ScheduledTask -TaskName $TaskName -TaskPath '\' -ErrorAction SilentlyContinue
            if ($null -ne $ownTask) { Unregister-ScheduledTask -InputObject $ownTask -Confirm:$false -ErrorAction Stop }
        } catch {
            $cleanupFailure = 'Could not unregister the one-shot task: ' + $_.Exception.Message
            if ($chainInitialized) {
                try { Write-ChainLog $cleanupFailure } catch { [Console]::Error.WriteLine($_.Exception.Message) }
            }
            [Console]::Error.WriteLine($cleanupFailure)
            $chainExit = 1
        }
    }
}
exit $chainExit
