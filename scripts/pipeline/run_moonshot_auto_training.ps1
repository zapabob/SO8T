<#
Moonshot 2025-2026 Training Automation (PowerShell)

Features:
 - Dataset download via HF CLI (optional)
 - Quadrality CoT dataset build (<think>...</thinking> switchable)
 - 5-minute rolling checkpoints (keep 3)
 - Auto-resume on power on (Task Scheduler)
 - Optional Discord/Slack notifications

Usage examples:
  # Run pipeline now (existing datasets)
  .\scripts\pipeline\run_moonshot_auto_training.ps1

  # Download datasets + run pipeline
  .\scripts\pipeline\run_moonshot_auto_training.ps1 -CollectNewData

  # Install auto-resume task (power on)
  .\scripts\pipeline\run_moonshot_auto_training.ps1 -InstallAutoResume

  # Uninstall auto-resume task
  .\scripts\pipeline\run_moonshot_auto_training.ps1 -UninstallAutoResume

  # Use </thinking> style + quadruple tokens + notify
  .\scripts\pipeline\run_moonshot_auto_training.ps1 -ThinkTagStyle openai -QuadrupleTokens `
    -DiscordWebhook "https://discord.com/api/webhooks/..." -NotifyOnStart -NotifyOnSuccess -NotifyOnFailure

  # Notify checkpoint + periodic summary (15 min, last 20 lines)
  .\scripts\pipeline\run_moonshot_auto_training.ps1 `
    -DiscordWebhook "https://discord.com/api/webhooks/..." `
    -NotifyCheckpoint -NotifySummaryMinutes 15 -NotifySummaryLines 20
#>

[CmdletBinding()]
param(
    [switch]$InstallAutoResume,
    [switch]$UninstallAutoResume,
    [switch]$CollectNewData,
    [switch]$Recover,
    [switch]$DryRun,
    [switch]$UseUnsloth,
    [switch]$McpApiSkill,
    [string]$TrainingConfig,
    [string]$SubagentStrategy = "parallel",
    [switch]$SubagentSchedule,
    [string]$GrapeVariant = "multiplicative",
    [switch]$EnableMhc,
    [switch]$EnableSo8,
    [string]$So8Mode = "mlp_only",
    [string]$MhcTargets = "o_proj,down_proj,up_proj,gate_proj",
    [string]$MhcBlend = "0.1",
    [string]$ThinkTagStyle,
    [switch]$QuadrupleTokens,
    [string]$DiscordWebhook,
    [string]$SlackWebhook,
    [switch]$NotifyOnStart,
    [switch]$NotifyOnSuccess,
    [switch]$NotifyOnFailure,
    [switch]$NotifyCheckpoint,
    [int]$NotifySummaryMinutes = 15,
    [int]$NotifySummaryLines = 20
)

Set-StrictMode -Version Latest

function Get-ProjectRoot {
    return (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
}

function Get-WebDatasetRoot {
    if (Test-Path "H:\from_D\webdataset") { return "H:\from_D\webdataset" }
    if (Test-Path "D:\webdataset") { return "D:\webdataset" }
    return (Join-Path (Get-ProjectRoot) "webdataset")
}

function Ensure-LogDir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Send-WebhookMessage {
    param(
        [string]$Message
    )
    if (-not $Message) { return }
    try {
        if ($DiscordWebhook) {
            $payload = @{ content = $Message } | ConvertTo-Json -Depth 4
            Invoke-RestMethod -Method Post -Uri $DiscordWebhook -Body $payload -ContentType "application/json" | Out-Null
        }
    } catch {
        Write-Host "[WARN] Discord notification failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    try {
        if ($SlackWebhook) {
            $payload = @{ text = $Message } | ConvertTo-Json -Depth 4
            Invoke-RestMethod -Method Post -Uri $SlackWebhook -Body $payload -ContentType "application/json" | Out-Null
        }
    } catch {
        Write-Host "[WARN] Slack notification failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Invoke-HfCliDownload {
    param(
        [string]$ProjectRoot
    )
    $webRoot = Get-WebDatasetRoot
    $baseDir = Join-Path $webRoot "hf_selected"
    $manifest = Join-Path $ProjectRoot "data\collected_2025_2026\hf_cli_manifest.json"
    $script = Join-Path $ProjectRoot "scripts\data_processing\hf_cli_dataset_fetch.py"

    Write-Host "[INFO] HF CLI download -> $baseDir" -ForegroundColor Cyan
    & py -3 $script --base-dir $baseDir --manifest $manifest
    if ($LASTEXITCODE -ne 0) {
        throw "HF CLI dataset fetch failed (exit code: $LASTEXITCODE)."
    }
}

function Invoke-QuadralityThinkBuild {
    param(
        [string]$ProjectRoot
    )
    $integratedDir = Join-Path $ProjectRoot "data\integrated"
    if (-not (Test-Path $integratedDir)) {
        Write-Host "[WARN] Integrated dataset directory not found: $integratedDir" -ForegroundColor Yellow
        return
    }
    $latest = Get-ChildItem -Path $integratedDir -Filter "*.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        Write-Host "[WARN] No integrated JSONL found for quadrality build." -ForegroundColor Yellow
        return
    }

    $builder = Join-Path $ProjectRoot "scripts\data_processing\build_quadrality_think_dataset.py"
    $output = Join-Path $integratedDir "quadrality_think.jsonl"

    $builderArgs = @("--input", $latest.FullName, "--output", $output)
    if ($env:SO8T_QUADRUPLE_TOKENS -eq "1") { $builderArgs += "--quadruple" }
    if ($env:SO8T_THINK_TAG_STYLE) { $builderArgs += @("--think-tag-style", $env:SO8T_THINK_TAG_STYLE) }

    Write-Host "[INFO] Quadrality CoT build from $($latest.Name)" -ForegroundColor Cyan
    & py -3 $builder @builderArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Quadrality <think> dataset build failed (exit code: $LASTEXITCODE)."
    }
}

function Register-AutoResumeTask {
    param(
        [string]$ProjectRoot,
        [string]$ScriptPath
    )
    $taskName = "SO8T-Moonshot-2025-2026-AutoResume"
    $taskArgs = "-ExecutionPolicy Bypass -File `"$ScriptPath`" -Recover"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $taskArgs -WorkingDirectory $ProjectRoot
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $trigger.Delay = "PT30S"
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false | Out-Null
    }

    Register-ScheduledTask -TaskName $taskName -Description "SO8T Moonshot 2025-2026 auto resume" `
        -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest | Out-Null

    Write-Host "[SUCCESS] Auto-resume task registered: $taskName" -ForegroundColor Green
}

function Unregister-AutoResumeTask {
    $taskName = "SO8T-Moonshot-2025-2026-AutoResume"
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false | Out-Null
        Write-Host "[SUCCESS] Auto-resume task removed: $taskName" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Auto-resume task not found: $taskName" -ForegroundColor Yellow
    }
}

$projectRoot = Get-ProjectRoot
$scriptPath = Join-Path $projectRoot "scripts\pipeline\run_moonshot_auto_training.ps1"

if ($InstallAutoResume) {
    Register-AutoResumeTask -ProjectRoot $projectRoot -ScriptPath $scriptPath
    return
}

if ($UninstallAutoResume) {
    Unregister-AutoResumeTask
    return
}

Set-Location $projectRoot

# Defaults from environment (optional)
if (-not $ThinkTagStyle -and $env:SO8T_THINK_TAG_STYLE) { $ThinkTagStyle = $env:SO8T_THINK_TAG_STYLE }
if (-not $DiscordWebhook -and $env:SO8T_DISCORD_WEBHOOK) { $DiscordWebhook = $env:SO8T_DISCORD_WEBHOOK }
if (-not $SlackWebhook -and $env:SO8T_SLACK_WEBHOOK) { $SlackWebhook = $env:SO8T_SLACK_WEBHOOK }

$notifyExplicit = $PSBoundParameters.ContainsKey("NotifyOnStart") -or `
    $PSBoundParameters.ContainsKey("NotifyOnSuccess") -or `
    $PSBoundParameters.ContainsKey("NotifyOnFailure") -or `
    $PSBoundParameters.ContainsKey("NotifyCheckpoint") -or `
    $PSBoundParameters.ContainsKey("NotifySummaryMinutes") -or `
    $PSBoundParameters.ContainsKey("NotifySummaryLines")
if (-not $notifyExplicit -and ($DiscordWebhook -or $SlackWebhook)) {
    $NotifyOnStart = $true
    $NotifyOnSuccess = $true
    $NotifyOnFailure = $true
    $NotifyCheckpoint = $true
}

# Environment defaults for 5-min checkpoints with 3 rolling slots
$env:SO8T_CHECKPOINT_INTERVAL = "300"
$env:SO8T_ROLLING_CHECKPOINTS = "3"
$env:SO8T_SUBAGENT_STRATEGY = $SubagentStrategy
$env:SO8T_SUBAGENT_SCHEDULE = "1"
$env:SO8T_GRAPE_VARIANT = $GrapeVariant
if ($ThinkTagStyle) { $env:SO8T_THINK_TAG_STYLE = $ThinkTagStyle }
if ($QuadrupleTokens) { $env:SO8T_QUADRUPLE_TOKENS = "1" }

if ($env:SO8T_NOTIFY_SUMMARY_MINUTES -and -not $PSBoundParameters.ContainsKey("NotifySummaryMinutes")) {
    [int]$NotifySummaryMinutes = $env:SO8T_NOTIFY_SUMMARY_MINUTES
}
if ($env:SO8T_NOTIFY_SUMMARY_LINES -and -not $PSBoundParameters.ContainsKey("NotifySummaryLines")) {
    [int]$NotifySummaryLines = $env:SO8T_NOTIFY_SUMMARY_LINES
}

if ($Recover) {
    $env:SO8T_RECOVER = "1"
    $env:SO8T_STARTUP_REGISTER = "0"
}
if ($UseUnsloth) { $env:SO8T_USE_UNSLOTH = "1" }
if ($McpApiSkill) { $env:SO8T_MCP_API_SKILL = "1" }
if ($EnableMhc) { $env:SO8T_MHC_ENABLE = "1" }
if ($EnableSo8) { $env:SO8T_SO8_ENABLE = "1" }
if ($So8Mode) { $env:SO8T_SO8_MODE = $So8Mode }
if ($MhcTargets) { $env:SO8T_MHC_TARGETS = $MhcTargets }
if ($MhcBlend) { $env:SO8T_MHC_BLEND = $MhcBlend }

$env:PYTHONPATH = "$projectRoot;$projectRoot\so8t-mmllm\src;$env:PYTHONPATH"

$logDir = Join-Path $projectRoot "logs\pipeline"
Ensure-LogDir -Path $logDir
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "moonshot_auto_training_$timestamp.log"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "SO8T Moonshot 2025-2026 Automation (PowerShell)" -ForegroundColor Cyan
Write-Host "Log: $logPath" -ForegroundColor Gray
Write-Host "=============================================" -ForegroundColor Cyan

if ($NotifyOnStart) {
    $tagStyle = if ($env:SO8T_THINK_TAG_STYLE) { $env:SO8T_THINK_TAG_STYLE } else { "legacy" }
    $quad = if ($env:SO8T_QUADRUPLE_TOKENS -eq "1") { "on" } else { "off" }
    $startMessage = "[SO8T] Moonshot pipeline start (2025-2026)`nRecover: $Recover`nCollectNewData: $CollectNewData`nThinkTagStyle: $tagStyle`nQuadrupleTokens: $quad`nNotifySummaryMinutes: $NotifySummaryMinutes`nLog: $logPath"
    Send-WebhookMessage -Message $startMessage
}

if ($CollectNewData) {
    Invoke-HfCliDownload -ProjectRoot $projectRoot
}

Invoke-QuadralityThinkBuild -ProjectRoot $projectRoot

$launcher = Join-Path $projectRoot "run_moonshot_pipeline_2025_2026.py"
$args = @("--use-existing-datasets")
if ($CollectNewData) { $args = @("--collect-new-data") }
if ($DryRun) { $args += "--dry-run" }
if ($UseUnsloth) { $args += "--use-unsloth" }
if ($McpApiSkill) { $args += "--mcp-api-skill" }
if ($Recover) { $args += "--recover" }
if ($TrainingConfig) { $args += @("--training-config", $TrainingConfig) }
if ($SubagentStrategy) { $args += @("--subagent-strategy", $SubagentStrategy) }
if ($SubagentSchedule) { $args += "--subagent-schedule" }
if ($EnableMhc) { $args += "--enable-mhc" }
if ($EnableSo8) { $args += "--enable-so8" }
if ($So8Mode) { $args += @("--so8-mode", $So8Mode) }
if ($MhcTargets) { $args += @("--mhc-targets", $MhcTargets) }
if ($MhcBlend) { $args += @("--mhc-blend", $MhcBlend) }
if ($GrapeVariant) { $args += @("--grape-variant", $GrapeVariant) }

Write-Host "[INFO] Launching: py -3 $launcher $($args -join ' ')" -ForegroundColor Green

$recentLines = New-Object System.Collections.Generic.Queue[string]
$lastSummary = Get-Date
$lastCheckpointNotice = Get-Date.AddMinutes(-999)

& py -3 $launcher @args 2>&1 | ForEach-Object {
    $line = $_
    if ($null -ne $line) {
        Add-Content -Path $logPath -Value $line -Encoding UTF8
        Write-Host $line
        $recentLines.Enqueue($line)
        while ($recentLines.Count -gt $NotifySummaryLines) {
            [void]$recentLines.Dequeue()
        }
        $now = Get-Date

        if ($NotifyCheckpoint -and $line -match "(?i)checkpoint" -and $line -match "\\.pt") {
            if (($now - $lastCheckpointNotice).TotalSeconds -gt 30) {
                $message = "[SO8T] Checkpoint saved`n$line`nLog: $logPath"
                Send-WebhookMessage -Message $message
                $lastCheckpointNotice = $now
            }
        }

        if ($NotifySummaryMinutes -gt 0 -and ($now - $lastSummary).TotalMinutes -ge $NotifySummaryMinutes) {
            if ($recentLines.Count -gt 0) {
                $summary = ($recentLines.ToArray() -join "`n")
                $message = "[SO8T] Progress summary (last $NotifySummaryLines lines)`n$summary`nLog: $logPath"
                Send-WebhookMessage -Message $message
                $lastSummary = $now
            }
        }
    }
}

$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host "[ERROR] Pipeline failed (exit code: $exitCode)." -ForegroundColor Red
    if ($NotifyOnFailure) {
        $failMessage = "[SO8T] Moonshot pipeline FAILED (exit code: $exitCode)`nLog: $logPath"
        Send-WebhookMessage -Message $failMessage
    }
    exit $exitCode
}

Write-Host "[SUCCESS] Pipeline completed." -ForegroundColor Green
if ($NotifyOnSuccess) {
    $successMessage = "[SO8T] Moonshot pipeline completed successfully.`nLog: $logPath"
    Send-WebhookMessage -Message $successMessage
}
