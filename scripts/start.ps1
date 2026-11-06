[CmdletBinding()]
param(
    [Alias('WithVision')][switch]$Vision = $true,
    [switch]$DownloadModels = $true,
    [switch]$SkipVision,
    [switch]$SkipModels
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$installVision = $Vision -and !$SkipVision
$downloadMissingModels = $DownloadModels -and !$SkipModels
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runtime = Join-Path $root '.runtime'
$statePath = Join-Path $runtime 'processes.json'
$server = Join-Path $root 'apps/server'
$web = Join-Path $root 'apps/web'
$state = @{ Root = $root; Processes = @(); Container = $null }

function Save-State {
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Invoke-Checked {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed (exit $LASTEXITCODE)." }
}

function Invoke-Docker {
    param([string[]]$Arguments)
    # Windows PowerShell can turn native stderr into terminating errors.
    # Check the native exit code explicitly so an offline engine permits fallback.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $docker.Source @Arguments 2>> (Join-Path $runtime 'mqtt.err.log'))
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    return @{ Output = $output; ExitCode = $exitCode }
}

function Start-Owned {
    param([string]$Name, [string]$File, [string]$Arguments, [string]$Directory)
    $child = Start-Process -FilePath $File -ArgumentList $Arguments -WorkingDirectory $Directory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtime "$Name.out.log") -RedirectStandardError (Join-Path $runtime "$Name.err.log")
    $identity = Get-CimInstance Win32_Process -Filter "ProcessId = $($child.Id)"
    if (!$identity) { throw "$Name exited during launch. Check .runtime/$Name.err.log." }
    $state.Processes += @{
        Name = $Name; Id = $child.Id; Created = $identity.CreationDate.ToUniversalTime().ToString('o')
        Executable = $identity.ExecutablePath; CommandLine = $identity.CommandLine
    }
    Save-State
    return $child.Id
}

function Test-Port {
    param([int]$Port)
    $client = New-Object Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (!$pending.AsyncWaitHandle.WaitOne(400)) { return $false }
        $client.EndConnect($pending)
        return $true
    } catch { return $false } finally { $client.Dispose() }
}

function Save-Descendants {
    param([int]$OwnerId, [string]$Name)
    $all = @(Get-CimInstance Win32_Process)
    $parents = @($OwnerId)
    do {
        $children = @($all | Where-Object { $_.ParentProcessId -in $parents -and $_.ProcessId -notin @($state.Processes | ForEach-Object { $_.Id }) })
        foreach ($child in $children) {
            $state.Processes += @{
                Name = "$Name child"; Id = $child.ProcessId; Created = $child.CreationDate.ToUniversalTime().ToString('o')
                Executable = $child.ExecutablePath; CommandLine = $child.CommandLine
            }
        }
        $parents = @($children | ForEach-Object { $_.ProcessId })
    } while ($parents.Count -gt 0)
    Save-State
}

function Test-OwnedListener {
    param([int]$ListenerId, [int]$OwnerId)
    $currentId = $ListenerId
    for ($depth = 0; $depth -lt 64 -and $currentId -gt 0; $depth++) {
        if ($currentId -eq $OwnerId) { return $true }
        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $currentId"
        if (!$current -or $current.ParentProcessId -eq $current.ProcessId) { return $false }
        $currentId = [int]$current.ParentProcessId
    }
    return $false
}

function Wait-OwnedPort {
    param([int]$Port, [int]$OwnerId, [string]$Name)
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if (!(Get-Process -Id $OwnerId -ErrorAction SilentlyContinue)) { throw "$Name exited. Check .runtime logs." }
        if (Test-Port $Port) {
            $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
            foreach ($listener in $listeners) {
                if (Test-OwnedListener ([int]$listener.OwningProcess) $OwnerId) { return }
            }
            if ($listeners.Count -eq 0) {
                Start-Sleep -Milliseconds 100
                continue
            }
            $owners = (@($listeners | ForEach-Object { $_.OwningProcess }) -join ', ')
            throw "Port $Port belongs to another process; $Name was not confirmed running (owner(s): $owners, expected root PID: $OwnerId)."
        }
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not listen on port $Port. Check .runtime logs."
}

# Serialize lifecycle operations across PowerShell sessions for this checkout.
$hash = [Security.Cryptography.SHA256]::Create()
try { $key = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant()))).Replace('-', '') } finally { $hash.Dispose() }
$mutex = New-Object Threading.Mutex($false, "Local\CamS-$key")
$locked = $false
$started = $false
try {
    try { $locked = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $locked = $true }
    if (!$locked) { throw 'Another lifecycle operation is in progress.' }
    if (Test-Path -LiteralPath $statePath) { throw 'Runtime state already exists. Run scripts/stop.ps1 before starting again.' }
    foreach ($required in @('apps/server/requirements-portal.txt', 'apps/server/src/portal_main.py', 'apps/web/package.json')) {
        if (!(Test-Path -LiteralPath (Join-Path $root $required))) { throw "Missing integration file: $required" }
    }
    if ($installVision -and !(Test-Path -LiteralPath (Join-Path $server 'requirements-vision.txt'))) { throw 'Missing apps/server/requirements-vision.txt; use -SkipVision for dashboard-only setup.' }
    if ($downloadMissingModels -and !(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'download_models.py'))) { throw 'Missing scripts/download_models.py; use -SkipModels if models are already provisioned.' }
    $uv = (Get-Command uv.exe -ErrorAction SilentlyContinue)
    if (!$uv) { $uv = (Get-Command uv -ErrorAction SilentlyContinue) }
    if (!$uv) {
        throw "No se encontró uv. Instálalo desde https://docs.astral.sh/uv/getting-started/installation/."
    }
    $uv = $uv.Source
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    # Keep installer caches and temporary files on the project drive.
    $env:UV_CACHE_DIR = Join-Path $runtime 'uv-cache'
    $env:TEMP = Join-Path $runtime 'install-temp'
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
    Invoke-Checked $uv @('--version')
    foreach ($port in @(8765, 5173)) { if (Test-Port $port) { throw "Port $port is already in use." } }
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    if (!(Test-Path -LiteralPath (Join-Path $root '.env'))) {
        Copy-Item -LiteralPath (Join-Path $root '.env.example') -Destination (Join-Path $root '.env')
    }
    Save-State
    $started = $true
    $venv = Join-Path $root '.venv-portal'
    $python = Join-Path $venv 'Scripts/python.exe'
    if (!(Test-Path -LiteralPath $python)) { Invoke-Checked $uv @('venv', '--python', '3.12', $venv) }
    Invoke-Checked $uv @('pip', 'install', '--python', $python, '-r', (Join-Path $server 'requirements-portal.txt'), '-c', (Join-Path $server 'requirements-portal.constraints.txt'))
    if ($installVision) {
        Invoke-Checked $uv @('pip', 'install', '--python', $python, 'cmake')
        $env:Path = (Join-Path $venv 'Scripts') + ';' + $env:Path
        $localWheels = Join-Path $runtime 'wheels'
        New-Item -ItemType Directory -Path $localWheels -Force | Out-Null
        Invoke-Checked $uv @('pip', 'install', '--python', $python, '--find-links', $localWheels, '-r', (Join-Path $server 'requirements-vision.txt'), '-c', (Join-Path $server 'requirements-portal.constraints.txt'))
        Invoke-Checked $python @('-c', 'import mediapipe, dlib, face_recognition')
    }
    if ($downloadMissingModels) { Invoke-Checked $python @((Join-Path $PSScriptRoot 'download_models.py')) }
    Push-Location $web
    try { Invoke-Checked $npm @('install', '--no-audit', '--no-fund') } finally { Pop-Location }

    $mqtt = Test-Port 1883
    if ($mqtt) { Write-Warning 'Port 1883 is already listening; using the existing broker. This script does not own it.' }
    if (!$mqtt) {
        $docker = Get-Command docker.exe -ErrorAction SilentlyContinue
        if ($docker) {
          try {
            $dockerInfo = Invoke-Docker @('info')
            $dockerInfo.Output | Set-Content -LiteralPath (Join-Path $runtime 'docker-info.log')
            if ($dockerInfo.ExitCode -eq 0) {
                $composeArgs = @('compose', '--project-directory', $root, '-f', (Join-Path $root 'docker-compose.yaml'))
                $previous = Invoke-Docker ($composeArgs + @('ps', '-a', '-q', 'mosquitto'))
                if ($previous.ExitCode -eq 0 -and @($previous.Output | Where-Object { $_ }).Count -eq 0) {
                    $composeStart = Invoke-Docker ($composeArgs + @('up', '-d', '--no-deps', 'mosquitto'))
                    $composeStart.Output | Set-Content -LiteralPath (Join-Path $runtime 'mqtt-compose.log')
                    $containers = Invoke-Docker ($composeArgs + @('ps', '-a', '-q', 'mosquitto'))
                    $containerIds = @($containers.Output)
                    if ($containers.ExitCode -eq 0 -and $containerIds.Count -eq 1 -and $containerIds[0]) {
                        $inspection = Invoke-Docker @('inspect', $containerIds[0])
                        if ($inspection.ExitCode -ne 0) { throw 'Cannot inspect the new Mosquitto container.' }
                        $details = @($inspection.Output | ConvertFrom-Json)[0]
                        $state.Container = @{ Id = $details.Id; Project = $details.Config.Labels.'com.docker.compose.project'; Service = 'mosquitto'; Directory = $root }
                        Save-State
                    }
                    if ($composeStart.ExitCode -ne 0) { Write-Warning 'Compose could not start Mosquitto. See .runtime/mqtt.err.log.' }
                    for ($i = 0; $i -lt 20 -and !$mqtt; $i++) { $mqtt = Test-Port 1883; if (!$mqtt) { Start-Sleep -Milliseconds 500 } }
                } else { Write-Warning 'An existing Compose broker was found or Compose is unavailable; it was not modified.' }
            } else { Write-Warning 'Docker is installed but its engine is unavailable.' }
          } catch { Write-Warning "Docker MQTT startup failed: $_" }
        }
        if (!$mqtt) {
            $mosquitto = Get-Command mosquitto.exe -ErrorAction SilentlyContinue
            $mosquittoPath = $null
            if ($mosquitto) { $mosquittoPath = $mosquitto.Source }
            if (!$mosquitto) {
                $candidate = Join-Path $env:ProgramFiles 'mosquitto/mosquitto.exe'
                if (Test-Path -LiteralPath $candidate) { $mosquittoPath = $candidate }
            }
            if ($mosquittoPath) {
                try {
                    # Default local-only listener; the Compose config uses container filesystem paths.
                    $brokerId = Start-Owned 'mqtt' $mosquittoPath '-p 1883' $root
                    Wait-OwnedPort 1883 $brokerId 'Mosquitto'
                    $mqtt = $true
                } catch { Write-Warning "Local Mosquitto failed: $_" }
            }
        }
    }
    if (!$mqtt) { Write-Warning 'MQTT IS UNAVAILABLE. The dashboard can start, but live MQTT events will not work. Install/start Docker or Mosquitto separately and retry.' }
    # portal_main loads the root .env itself. Pin the local development endpoints.
    $savedEnv = @{}
    foreach ($name in @('SERVER_HOST', 'SERVER_PORT', 'MQTT_BROKER_ADDRESS', 'MQTT_BROKER_PORT', 'SERIAL_PORT', 'CAMS_SERIAL_BAUD')) { $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:SERVER_HOST = '127.0.0.1'
        $env:SERVER_PORT = '8765'
        $env:MQTT_BROKER_ADDRESS = '127.0.0.1'
        $env:MQTT_BROKER_PORT = '1883'
        $env:SERIAL_PORT = 'COM3'
        $env:CAMS_SERIAL_BAUD = '9600'
        $backendId = Start-Owned 'backend' $python '-m src.portal_main' $server
    } finally { foreach ($name in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnv[$name], 'Process') } }
    Wait-OwnedPort 8765 $backendId 'Backend'
    Save-Descendants $backendId 'backend'
    $webId = Start-Owned 'web' $env:ComSpec ('/d /s /c ""{0}" run dev -- --host 127.0.0.1 --port 5173 --strictPort"' -f $npm) $web
    Wait-OwnedPort 5173 $webId 'React dev server'
    Save-Descendants $webId 'web'
    Write-Host 'Dashboard: http://127.0.0.1:5173 | Backend: http://127.0.0.1:8765'
    Write-Host "Logs and process identities: $runtime"
    Write-Host "MQTT listening on 127.0.0.1:1883: $mqtt (TCP check; application connection is reported by the backend)."
} catch {
    if ($started) {
        try { & (Join-Path $PSScriptRoot 'stop.ps1') } catch { Write-Warning "Cleanup incomplete: $_" }
    }
    throw
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
