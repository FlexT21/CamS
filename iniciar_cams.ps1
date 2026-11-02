[CmdletBinding()]
param([switch]$InstalarVision)
$ErrorActionPreference = 'Stop'
if ($InstalarVision) {
    & (Join-Path $PSScriptRoot 'scripts/start.ps1')
} else {
    & (Join-Path $PSScriptRoot 'scripts/start.ps1') -SkipVision -SkipModels
}
