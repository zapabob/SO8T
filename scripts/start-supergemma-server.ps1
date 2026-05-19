#Requires -Version 5.1
<#
.SYNOPSIS
    SuperGemma4 E4B llama-server launcher for RTX 3060 (12GB VRAM) — TurboQuant build
.DESCRIPTION
    Adapted from Qwen3.6-35B reference server flags where safe for dense ~4B Q8_0 on 12GB VRAM.
    - Model   : supergemma4-e4b-abliterated Q8_0 (7.48 GB)
    - Server  : llama-turboquant v9458 (31b900be6)
    - API     : http://<BindHost>:8080/v1  (OpenAI-compatible)
.PARAMETER Profile
    KV cache profile: Turbo=turbo4 (default, ~40% less KV VRAM), Stable=q8_0.
.PARAMETER Lan
    Listen on 0.0.0.0 (all interfaces). Default is 127.0.0.1 only.
.PARAMETER BindHost
    Override listen address (ignored if -Lan is set).
.PARAMETER ContextSize
    Context window (-c / --ctx-size). Default 100000 per user request.
    WARNING: 100k + Q8_0 (~7.5GB) on RTX 3060 12GB will likely OOM. Use -ContextSize 8192 for safe local use.
.PARAMETER GpuLayers
    GPU layers (-ngl). Default 99 (all on GPU). Lower (e.g. 28) if OOM with large context.
.PARAMETER AutoFallbackContext
    If startup fails or VRAM is tight, retry with 32768 -> 16384 -> 8192 (logged).
.PARAMETER DryRun
    Validate paths and print command without starting the server.
.EXAMPLE
    .\start-supergemma-server.ps1
    .\start-supergemma-server.ps1 -ContextSize 8192
    .\start-supergemma-server.ps1 -Profile Stable -ContextSize 16384
    .\start-supergemma-server.ps1 -Lan -DryRun
.NOTES
    Reference (NOT used as-is): Qwen3.6-35B MoE -ncmoe 30 -c 32768 -ngl 999 on larger GPU.
    Hermes: OPENAI_API_BASE=http://127.0.0.1:8080/v1
#>
param(
    [ValidateSet("Turbo", "Stable")]
    [string]$Profile = "Turbo",
    [int]$ContextSize = 100000,
    [int]$GpuLayers = 99,
    [switch]$AutoFallbackContext,
    [switch]$Lan,
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8080,
    [switch]$DryRun
)

if ($Lan) { $BindHost = "0.0.0.0" }

# Stable profile: safer KV (q8_0) + smaller default context unless user set -ContextSize
if ($Profile -eq "Stable" -and -not $PSBoundParameters.ContainsKey("ContextSize")) {
    $ContextSize = 8192
}

# ── Paths ───────────────────────────────────────────────────────────────────
$BIN      = "C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe"
$MODEL    = "C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf"
$MODEL_TQ = "C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.tq4_1s.gguf"
$API_KEY  = "llama-cpp-local"
$LOG_DIR  = "C:\Users\downl\Desktop\SO8T\logs"
$LOG_FILE = "$LOG_DIR\llama-server-supergemma4.log"

# ── RTX 3060 tuning (VRAM math) ─────────────────────────────────────────────
# Model Q8_0 on disk:     ~7.48 GB  -> ~7,680 MiB VRAM weights (-ngl 99)
# KV scales ~linearly with -c (ctx-size). Rough E4B dense estimate on GPU:
#   turbo4 @ 8192:   ~0.6 GB  |  turbo4 @ 100000: ~7+ GB  (often OOM with 7.5GB weights)
#   q8_0   @ 8192:   ~1.0 GB  |  q8_0   @ 16384:  ~2.0 GB (safer max on 12GB)
# Weights Q8_0 -ngl 99: ~7.5 GB | partial -ngl: weights split CPU/GPU, more room for KV
# RTX 3060 recommended: -ContextSize 8192 (turbo4) or 16384 (aggressive); 100000 needs >>12GB VRAM
# MoE -ncmoe: N/A (SuperGemma4 is dense)
$SCRIPT:SAFE_CONTEXT_PRESETS = @(4096, 8192, 16384, 32768)
$MAX_PREDICT  = 4096
$CPU_THREADS  = 6
$BATCH_SIZE   = 512
$UBATCH_SIZE  = 512
$PARALLEL     = 1

function Test-LlamaKvTypeSupported {
    param([string]$BinPath, [string]$KvType)
    $help = & $BinPath --help 2>&1 | Out-String
    return ($help -match 'turbo2,\s*turbo3,\s*turbo4') -and ($help -match $KvType)
}

function Resolve-KvCacheType {
    param(
        [string]$BinPath,
        [string]$RequestedProfile
    )
    $map = @{ Turbo = "turbo4"; Stable = "q8_0" }
    $desired = $map[$RequestedProfile]
    if ($RequestedProfile -eq "Turbo") {
        if (Test-LlamaKvTypeSupported -BinPath $BinPath -KvType "turbo4") {
            return @{ Type = "turbo4"; Profile = "Turbo"; Fallback = $false }
        }
        Write-Warn "turbo4 not in binary help; falling back to Stable (q8_0)"
        return @{ Type = "q8_0"; Profile = "Stable (fallback)"; Fallback = $true }
    }
    return @{ Type = "q8_0"; Profile = "Stable"; Fallback = $false }
}

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan    }
function Write-OK($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green   }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow  }
function Write-Fail($msg)  { Write-Host "[FAIL]  $msg" -ForegroundColor Red     }

function Get-ContextFallbackChain {
    param([int]$Requested)
    $chain = @($Requested) + @(32768, 16384, 8192, 4096) | Select-Object -Unique
    return $chain
}

function Test-LlamaServerHealthy {
    param([int]$PortNum = 8080, [int]$TimeoutSec = 3)
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:${PortNum}/health" -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

function Start-LlamaBackgroundProbe {
    param(
        [string]$BinPath,
        [string[]]$Args,
        [string]$LogPath,
        [int]$WaitSec = 90
    )
    if (Test-Path $LogPath) { Remove-Item $LogPath -Force }
    $errLog = "${LogPath}.err"
    if (Test-Path $errLog) { Remove-Item $errLog -Force }
    $proc = Start-Process -FilePath $BinPath -ArgumentList $Args `
        -RedirectStandardOutput $LogPath -RedirectStandardError $errLog `
        -WindowStyle Hidden -PassThru
    $ok = $false
    $oom = $false
    for ($t = 0; $t -lt $WaitSec; $t += 3) {
        Start-Sleep -Seconds 3
        if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
            $oom = $true
            break
        }
        $blob = ""
        if (Test-Path $LogPath) { $blob += Get-Content $LogPath -Raw -ErrorAction SilentlyContinue }
        if (Test-Path $errLog) { $blob += Get-Content $errLog -Raw -ErrorAction SilentlyContinue }
        if ($blob -match 'out of memory|CUDA error|failed to allocate|not enough memory') {
            $oom = $true
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            break
        }
        if (Test-LlamaServerHealthy) { $ok = $true; break }
    }
    if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    return @{ Ok = $ok; OOM = $oom; Log = $LogPath }
}

function Show-ContextVramWarning {
    param([int]$Ctx, [string]$KvType, [int]$Ngl)
    if ($Ctx -le 16384) { return }
    Write-Warn "ContextSize=$Ctx is very large for RTX 3060 12GB + Q8_0 (~7.5GB weights)."
    Write-Warn "KV type=$KvType, -ngl $Ngl. Expect OOM or auto-fit shrink unless layers offloaded to CPU."
    Write-Warn "Safe presets on 12GB: $($SCRIPT:SAFE_CONTEXT_PRESETS -join ', ') (with -Profile Turbo)."
    if ($Ctx -ge 100000) {
        Write-Warn "100k context typically needs 24GB+ VRAM for this model size, or heavy CPU KV (-ngl partial)."
    }
}

function Get-SuperGemmaServerArgs {
    param(
        [string]$ModelPath,
        [string]$KvType
    )
    return @(
        "-m", $ModelPath,
        "--host", $BindHost,
        "--port", $Port,
        "--api-key", $API_KEY,
        "-ngl", $GpuLayers,
        "-c", $ContextSize,
        "-n", $MAX_PREDICT,
        "-t", $CPU_THREADS,
        "-b", $BATCH_SIZE,
        "-ub", $UBATCH_SIZE,
        "-np", $PARALLEL,
        "-cb",
        "-fa", "on",
        "--cache-type-k", $KvType,
        "--cache-type-v", $KvType,
        "--reasoning", "off",
        "--no-cache-prompt",
        "--checkpoint-every-n-tokens", "-1",
        "--jinja",
        "--metrics"
    )
}

Write-Host ""
Write-Host "=== SuperGemma4 llama-server | RTX 3060 | TurboQuant ===" -ForegroundColor Magenta
Write-Host ""

if ($BindHost -eq "0.0.0.0") {
    Write-Warn "BindHost=0.0.0.0: LAN-wide exposure. Ensure firewall + api-key are set."
} else {
    Write-OK "BindHost=$BindHost (localhost-only)"
}

if (-not (Test-Path $BIN)) {
    Write-Fail "llama-server.exe not found: $BIN"
    exit 1
}
Write-OK "Binary: $BIN"

$kvResolved = Resolve-KvCacheType -BinPath $BIN -RequestedProfile $Profile
$KV_TYPE = $kvResolved.Type
Write-OK "KV profile: $($kvResolved.Profile) -> --cache-type-k/v $KV_TYPE"
Write-OK "Context: -c $ContextSize (--ctx-size) | GPU layers: -ngl $GpuLayers"
Show-ContextVramWarning -Ctx $ContextSize -KvType $KV_TYPE -Ngl $GpuLayers

$ModelPath = $MODEL
if (-not (Test-Path $ModelPath)) {
    Write-Warn "Q8_0 model not found, trying TQ4_1S variant..."
    if (Test-Path $MODEL_TQ) {
        $ModelPath = $MODEL_TQ
        Write-OK "Fallback TQ4_1S: $ModelPath (6.43 GB)"
    } else {
        Write-Fail "Model not found: $MODEL"
        exit 1
    }
} else {
    Write-OK "Model: $ModelPath (7.48 GB Q8_0)"
}

$nvidiasmi = & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>&1
if ($LASTEXITCODE -eq 0) {
    $parts = $nvidiasmi -split ","
    $free = [int]$parts[3].Trim()
    Write-OK "GPU: $($parts[0].Trim()) | Free: ${free}MiB"
    if ($free -lt 7500 -and -not $DryRun) {
        Write-Warn "VRAM low (${free}MiB). Stop other GPU jobs or use -c 4096."
        Start-Sleep 5
    }
} else {
    Write-Warn "nvidia-smi failed"
}

if (-not $DryRun) {
    $portCheck = netstat -ano 2>&1 | Select-String ":${Port}\s" | Select-String "LISTENING"
    if ($portCheck) {
        $procPid = ($portCheck | Select-Object -First 1 -ExpandProperty Line) -split '\s+' | Select-Object -Last 1
        try {
            $proc = Get-Process -Id $procPid -ErrorAction Stop
            if ($proc.Name -like "*llama*") {
                Write-Info "Stopping existing llama-server (PID $procPid)..."
                Stop-Process -Id $procPid -Force
                Start-Sleep 2
            } else {
                Write-Fail "Port $Port in use by $($proc.Name)"
                exit 1
            }
        } catch {
            Write-Warn "Could not inspect PID on port $Port"
        }
    }
}

if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
}

$effectiveContext = $ContextSize
$contextChain = if ($AutoFallbackContext) { Get-ContextFallbackChain -Requested $ContextSize } else { @($ContextSize) }

if ($AutoFallbackContext -and $ContextSize -ge 100000) {
    Write-Info "AutoFallbackContext enabled: will try $($contextChain -join ' -> ') if needed"
}

$chosenContext = $null
$probeLog = "$LOG_DIR\llama-server-probe.log"

if ($AutoFallbackContext -and -not $DryRun) {
    foreach ($tryCtx in $contextChain) {
        $ContextSize = $tryCtx
        Write-Info "Probe start with -c $tryCtx ..."
        $probeArgs = Get-SuperGemmaServerArgs -ModelPath $ModelPath -KvType $KV_TYPE
        $probe = Start-LlamaBackgroundProbe -BinPath $BIN -Args $probeArgs -LogPath $probeLog -WaitSec 120
        if ($probe.Ok) {
            $chosenContext = $tryCtx
            Write-OK "Probe OK at -c $tryCtx"
            break
        }
        Write-Warn "Probe failed at -c $tryCtx (OOM=$($probe.OOM))"
    }
    if (-not $chosenContext) {
        Write-Fail "All context sizes failed. RTX 3060 12GB max feasible (tested): ~32768 with turbo4 -ngl 99."
        Write-Fail "Try: -Profile Stable -ContextSize 8192  OR  -GpuLayers 28 -ContextSize 16384"
        exit 1
    }
    $ContextSize = $chosenContext
    $effectiveContext = $chosenContext
} else {
    $effectiveContext = $ContextSize
}

$ServerArgs = Get-SuperGemmaServerArgs -ModelPath $ModelPath -KvType $KV_TYPE
$cmdLine = "$BIN " + ($ServerArgs -join " ")

Write-Host ""
Write-Info "Command (-c $effectiveContext):"
Write-Host "  $cmdLine" -ForegroundColor DarkCyan
Write-Host ""
Write-Info "API: http://${BindHost}:${Port}/v1"
Write-Info "Metrics: http://${BindHost}:${Port}/metrics (Prometheus)"
Write-Info "Hermes: OPENAI_API_BASE=http://127.0.0.1:${Port}/v1"

if ($DryRun) {
    Write-OK "DryRun complete — server not started."
    exit 0
}

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$ts] === SuperGemma4 llama-server ===" | Tee-Object -FilePath $LOG_FILE -Append | Out-Null
"[$ts] BindHost=$BindHost Profile=$Profile ctx=$effectiveContext Model=$ModelPath" | Tee-Object -FilePath $LOG_FILE -Append | Out-Null
"[$ts] $cmdLine" | Tee-Object -FilePath $LOG_FILE -Append | Out-Null

Write-OK "Starting server (Ctrl+C to stop)..."
try {
    & $BIN @ServerArgs 2>&1 | Tee-Object -FilePath $LOG_FILE -Append
} catch {
    Write-Fail "Start error: $_"
    if (-not $AutoFallbackContext) {
        Write-Fail "Hint: re-run with -AutoFallbackContext or -ContextSize 8192 / -Profile Stable"
    }
    exit 1
}
