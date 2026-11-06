[CmdletBinding()]
param(
    [switch]$DiscardStaleState
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$statePath = Join-Path $root '.runtime/processes.json'
$hash = [Security.Cryptography.SHA256]::Create()
try { $key = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant()))).Replace('-', '') } finally { $hash.Dispose() }
$mutex = New-Object Threading.Mutex($false, "Local\CamS-$key")
$locked = $false

function Invoke-Docker {
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $docker.Source @Arguments 2>> (Join-Path $root '.runtime/mqtt.err.log'))
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    return @{ Output = $output; ExitCode = $exitCode }
}

function Get-UtcTicks {
    param($Value)
    # PS 5.1 leaves JSON dates as strings; PS 7 can deserialize DateTime values.
    if ($Value -is [datetimeoffset]) { return $Value.UtcTicks }
    if ($Value -is [datetime]) { return $Value.ToUniversalTime().Ticks }
    return [datetimeoffset]::Parse([string]$Value, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::AssumeUniversal).UtcTicks
}

function Same-Process {
    param($Live, $Saved)
    if ($null -eq $Live -or !$Live.ExecutablePath -or !$Live.CommandLine) { return $false }
    try {
        return ($Live.ExecutablePath -eq $Saved.Executable -and $Live.CommandLine -ceq $Saved.CommandLine -and (Get-UtcTicks $Live.CreationDate) -eq (Get-UtcTicks $Saved.Created))
    } catch { return $false }
}

function Stop-OwnedTree {
    param($Saved)
    $live = Get-CimInstance Win32_Process -Filter "ProcessId = $($Saved.Id)"
    if (!$live) { return }
    if (!(Same-Process $live $Saved)) { throw "Ownership mismatch for live PID $($Saved.Id) ($($Saved.Name)); refusing to stop it. Runtime state retained." }
    # Capture descendants while the parent identity is known. Validate each again
    # immediately before stopping it, including its executable and creation time.
    $all = @(Get-CimInstance Win32_Process)
    $tree = [Collections.Generic.List[object]]::new()
    $tree.Add($Saved)
    for ($index = 0; $index -lt $tree.Count; $index++) {
        $parent = $tree[$index]
        foreach ($child in @($all | Where-Object { $_.ParentProcessId -eq $parent.Id -and (Get-UtcTicks $_.CreationDate) -ge (Get-UtcTicks $parent.Created) })) {
            if ($child.ProcessId -in @($tree | ForEach-Object { $_.Id })) { continue }
            $tree.Add([pscustomobject]@{ Id = $child.ProcessId; Created = $child.CreationDate.ToUniversalTime().ToString('o'); Executable = $child.ExecutablePath; CommandLine = $child.CommandLine })
        }
    }
    for ($index = $tree.Count - 1; $index -ge 0; $index--) {
        $entry = $tree[$index]
        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($entry.Id)"
        if (!$current) { continue }
        if (!(Same-Process $current $entry)) { throw "Ownership mismatch for live PID $($entry.Id); runtime state retained." }
        try {
            $process = Get-Process -Id $entry.Id -ErrorAction Stop
            Stop-Process -InputObject $process -ErrorAction Stop
            if (!$process.WaitForExit(5000)) { throw "Timed out stopping PID $($entry.Id)." }
        } catch {
            if (Get-CimInstance Win32_Process -Filter "ProcessId = $($entry.Id)") { throw }
        }
    }
}

try {
    try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
    if (!$locked) { throw 'Another lifecycle operation is in progress.' }
    if (!(Test-Path -LiteralPath $statePath)) { Write-Host 'No managed runtime state found.'; return }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.Root -ne $root) { throw 'Runtime state belongs to a different checkout.' }
    $processes = @($state.Processes)
    # Validate all live entries before stopping anything. Mismatches must never
    # discard the only record of processes that still require investigation.
    $mismatches = @()
    $ownedLive = @()
    foreach ($entry in $processes) {
        $live = Get-CimInstance Win32_Process -Filter "ProcessId = $($entry.Id)"
        if ($live) {
            if (Same-Process $live $entry) {
                $ownedLive += $entry
            } else {
                $mismatches += $entry
            }
        }
    }
    if ($mismatches.Count -gt 0) {
        if ($DiscardStaleState -and $ownedLive.Count -eq 0 -and $null -eq $state.Container) {
            Remove-Item -LiteralPath $statePath
            Write-Host 'Stale runtime state discarded. No live managed process was stopped.'
            return
        }
        $firstMismatch = $mismatches[0]
        throw "Ownership mismatch for live PID $($firstMismatch.Id) ($($firstMismatch.Name)); runtime state retained. If all managed processes are already gone, run scripts/stop.ps1 -DiscardStaleState."
    }
    for ($index = $processes.Count - 1; $index -ge 0; $index--) { Stop-OwnedTree $processes[$index] }
    if ($state.Container) {
        $docker = Get-Command docker.exe -ErrorAction Stop
        $inspection = Invoke-Docker @('inspect', $state.Container.Id)
        if ($inspection.ExitCode -ne 0) {
            $dockerInfo = Invoke-Docker @('info')
            if ($dockerInfo.ExitCode -ne 0) { throw 'Docker engine unavailable; runtime state retained for a later stop.' }
            $knownIds = Invoke-Docker @('ps', '-a', '--no-trunc', '-q')
            if ($knownIds.ExitCode -ne 0 -or $state.Container.Id -in $knownIds.Output) { throw 'Cannot confirm container identity; runtime state retained.' }
        } else {
            $details = @($inspection.Output | ConvertFrom-Json)[0]
            $labels = $details.Config.Labels
            if ($details.Id -ne $state.Container.Id -or $labels.'com.docker.compose.project' -ne $state.Container.Project -or $labels.'com.docker.compose.service' -ne 'mosquitto' -or $labels.'com.docker.compose.project.working_dir' -ne $root) {
                throw 'Container identity mismatch; refusing to stop it. Runtime state retained.'
            }
            $stopped = Invoke-Docker @('stop', $details.Id)
            if ($stopped.ExitCode -ne 0) { throw 'Could not stop owned broker; runtime state retained.' }
        }
    }
    foreach ($entry in $processes) {
        if (Get-CimInstance Win32_Process -Filter "ProcessId = $($entry.Id)") { throw "PID $($entry.Id) is still present; runtime state retained." }
    }
    Remove-Item -LiteralPath $statePath
    Write-Host 'Managed processes stopped. Logs, virtual environment and broker data retained.'
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
