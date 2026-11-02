param([string]$ArduinoPort = "")

$ErrorActionPreference = "Stop"
if ($ArduinoPort -and $ArduinoPort -notmatch '^COM[0-9]+$') {
    throw "ArduinoPort debe ser un puerto como COM3"
}

$projectRoot = $PSScriptRoot
$serverRoot = Join-Path $projectRoot "apps\server"
$clientRoot = Join-Path $projectRoot "apps\client"
$serverPython = Join-Path $serverRoot ".venv312\Scripts\python.exe"
$clientPython = Join-Path $clientRoot ".venv312\Scripts\python.exe"
$mosquitto = "C:\Program Files\Mosquitto\mosquitto.exe"

if (-not (Test-Path $serverPython)) {
    throw "No se encontró el entorno del servidor: $serverPython"
}

if (-not (Test-Path $clientPython)) {
    throw "No se encontró el entorno del cliente: $clientPython"
}

$mqtt = Get-NetTCPConnection -LocalPort 1883 -State Listen -ErrorAction SilentlyContinue
if (-not $mqtt) {
    if (-not (Test-Path $mosquitto)) {
        throw "No se encontró Mosquitto en $mosquitto"
    }

    Start-Process -WindowStyle Hidden -FilePath $mosquitto -ArgumentList "-v" -WorkingDirectory (Split-Path $mosquitto)
    Start-Sleep -Seconds 2
}

if (-not (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -WindowStyle Hidden powershell.exe -WorkingDirectory $serverRoot -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-Command",
        "& '$serverPython' -m src.main"
    )
    Start-Sleep -Seconds 3
}

Start-Process -WindowStyle Hidden powershell.exe -WorkingDirectory $clientRoot -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-Command",
    "& '$clientPython' -m src.main 0 --server ws://localhost:8765/api/ws/ $(if ($ArduinoPort) { '--arduino-port ' + $ArduinoPort })"
)

Write-Host "CamS iniciado. Cierra la ventana de la cámara con Esc."
