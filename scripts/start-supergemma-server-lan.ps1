#Requires -Version 5.1
<#
.SYNOPSIS
    SuperGemma4 llama-server — LAN binding (0.0.0.0)
.DESCRIPTION
    Same flags as start-supergemma-server.ps1 but listens on all interfaces.
    Security: requires firewall rules; api-key still required for API calls.
.EXAMPLE
    .\start-supergemma-server-lan.ps1
    .\start-supergemma-server-lan.ps1 -DryRun
#>
param(
    [switch]$DryRun
)

$launcher = Join-Path $PSScriptRoot "start-supergemma-server.ps1"
if (-not (Test-Path $launcher)) {
    Write-Error "Missing launcher: $launcher"
    exit 1
}

$params = @{ Lan = $true }
if ($DryRun) { $params.DryRun = $true }
& $launcher @params
