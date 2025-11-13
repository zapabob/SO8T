# Borea-Phi-3.5 SO8T/thinking Complete Pipeline Script (PowerShell)
# Executes Steps 1-5 sequentially

Write-Host "[PIPELINE] Borea-Phi-3.5 SO8T/thinking Complete Pipeline" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""

# Step 1: Dataset creation (skip if already exists)
Write-Host "[STEP 1] Checking dataset..." -ForegroundColor Cyan
$DATASET = "D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl"

if (Test-Path $DATASET) {
    Write-Host "[OK] Dataset already exists: $DATASET" -ForegroundColor Green
} else {
    Write-Host "[STEP 1] Creating /think format dataset..." -ForegroundColor Yellow
    $files = Get-ChildItem "D:\webdataset\processed\four_class\four_class_*.jsonl" | Select-Object -ExpandProperty FullName
    py -3 scripts\data\create_thinking_sft_dataset.py --inputs $files --output $DATASET
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Step 1 failed" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Step 2: Training execution (fast or full mode)
Write-Host "[STEP 2] Starting training..." -ForegroundColor Cyan
$TRAINING_MODE = "fast"
if ($args[0] -eq "full") {
    $TRAINING_MODE = "full"
}

if ($TRAINING_MODE -eq "fast") {
    Write-Host "[INFO] Using fast training configuration" -ForegroundColor Yellow
    $CONFIG = "configs\train_borea_phi35_so8t_thinking_fast.yaml"
    $OUTPUT_DIR = "D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking_fast"
} else {
    Write-Host "[INFO] Using full training configuration" -ForegroundColor Yellow
    $CONFIG = "configs\train_borea_phi35_so8t_thinking.yaml"
    $OUTPUT_DIR = "D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking"
}

py -3 scripts\training\train_borea_phi35_so8t_thinking.py `
    --config $CONFIG `
    --dataset $DATASET `
    --output-dir $OUTPUT_DIR `
    --auto-resume

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Step 2 failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Baking process
Write-Host "[STEP 3] Baking SO8T rotations..." -ForegroundColor Cyan
$TRAINED_MODEL = Join-Path $OUTPUT_DIR "final_model"
$BAKED_MODEL = "D:\webdataset\borea_phi35_so8t_thinking\baked_model"

if (-not (Test-Path $TRAINED_MODEL)) {
    Write-Host "[ERROR] Trained model not found: $TRAINED_MODEL" -ForegroundColor Red
    exit 1
}

py -3 scripts\training\bake_borea_phi35_so8t.py `
    --model-path $TRAINED_MODEL `
    --output-path $BAKED_MODEL

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Step 3 failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 4: GGUF conversion
Write-Host "[STEP 4] Converting to GGUF format..." -ForegroundColor Cyan
$GGUF_OUTPUT_DIR = "D:\webdataset\gguf_models\borea_phi35_so8t_thinking"

if (-not (Test-Path $BAKED_MODEL)) {
    Write-Host "[ERROR] Baked model not found: $BAKED_MODEL" -ForegroundColor Red
    exit 1
}

py -3 scripts\conversion\convert_borea_so8t_to_gguf.py `
    --model-path $BAKED_MODEL `
    --output-dir $GGUF_OUTPUT_DIR `
    --model-name borea_phi35_so8t_thinking `
    --quantization-types f16 q8_0 q4_k_m

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Step 4 failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Ollama import and testing
Write-Host "[STEP 5] Importing to Ollama and testing..." -ForegroundColor Cyan
$MODELFILE = "modelfiles\borea_phi35_so8t_thinking.modelfile"
$GGUF_FILE = Join-Path $GGUF_OUTPUT_DIR "borea_phi35_so8t_thinking_Q8_0.gguf"

if (-not (Test-Path $GGUF_FILE)) {
    Write-Host "[ERROR] GGUF file not found: $GGUF_FILE" -ForegroundColor Red
    exit 1
}

# Update FROM path in Modelfile
$content = Get-Content $MODELFILE -Raw
$content = $content -replace 'FROM .*', "FROM $GGUF_FILE"
Set-Content $MODELFILE -Value $content -NoNewline

Write-Host "[INFO] Creating Ollama model..." -ForegroundColor Yellow
ollama create borea-phi35-so8t-thinking -f $MODELFILE

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Ollama import failed, but continuing..." -ForegroundColor Yellow
} else {
    Write-Host "[OK] Ollama model created" -ForegroundColor Green
    Write-Host ""
    Write-Host "[TEST] Testing /think format inference..." -ForegroundColor Cyan
    ollama run borea-phi35-so8t-thinking "Solve this problem. First organize your thinking steps, then provide the final answer. What is 2+2?"
}
Write-Host ""

Write-Host "[SUCCESS] Complete pipeline finished!" -ForegroundColor Green
Write-Host "[AUDIO] Playing completion notification..." -ForegroundColor Cyan
& "scripts\utils\play_audio_notification.ps1"



