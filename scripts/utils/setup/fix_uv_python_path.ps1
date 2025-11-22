# uv Python パス修正スクリプト
# Usage: .\scripts\utils\setup\fix_uv_python_path.ps1

Write-Host "[INFO] Fixing uv Python path detection..." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan

# Pythonインタープリターの検出
Write-Host "[STEP 1] Detecting Python interpreters..." -ForegroundColor Yellow

$pythonPaths = @()

# py launcher経由で検出
try {
    $pyPython = py -3 -c "import sys; print(sys.executable)" 2>&1
    if ($LASTEXITCODE -eq 0 -and $pyPython) {
        Write-Host "  [OK] Found via py launcher: $pyPython" -ForegroundColor Green
        $pythonPaths += $pyPython
    }
} catch {
    Write-Host "  [SKIP] py launcher not available" -ForegroundColor Yellow
}

# python コマンド経由で検出
try {
    $pythonCmd = python -c "import sys; print(sys.executable)" 2>&1
    if ($LASTEXITCODE -eq 0 -and $pythonCmd) {
        Write-Host "  [OK] Found via python command: $pythonCmd" -ForegroundColor Green
        if ($pythonPaths -notcontains $pythonCmd) {
            $pythonPaths += $pythonCmd
        }
    }
} catch {
    Write-Host "  [SKIP] python command not available" -ForegroundColor Yellow
}

# python3 コマンド経由で検出
try {
    $python3Cmd = python3 -c "import sys; print(sys.executable)" 2>&1
    if ($LASTEXITCODE -eq 0 -and $python3Cmd) {
        Write-Host "  [OK] Found via python3 command: $python3Cmd" -ForegroundColor Green
        if ($pythonPaths -notcontains $python3Cmd) {
            $pythonPaths += $python3Cmd
        }
    }
} catch {
    Write-Host "  [SKIP] python3 command not available" -ForegroundColor Yellow
}

if ($pythonPaths.Count -eq 0) {
    Write-Host "[ERROR] No Python interpreters found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.10+ and ensure it's in PATH" -ForegroundColor Yellow
    exit 1
}

# 最初に見つかったPythonを使用
$selectedPython = $pythonPaths[0]
Write-Host ""
Write-Host "[STEP 2] Selected Python interpreter:" -ForegroundColor Yellow
Write-Host "  $selectedPython" -ForegroundColor Green

# Python バージョン確認
Write-Host ""
Write-Host "[STEP 3] Verifying Python version..." -ForegroundColor Yellow
$version = & $selectedPython --version 2>&1
Write-Host "  $version" -ForegroundColor White

# uv でPythonパスを設定
Write-Host ""
Write-Host "[STEP 4] Testing uv with explicit Python path..." -ForegroundColor Yellow
$testResult = uv pip install --python $selectedPython --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] uv can use this Python interpreter" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] uv test failed" -ForegroundColor Yellow
    Write-Host $testResult -ForegroundColor Red
}

Write-Host ""
Write-Host "[INFO] Usage with explicit Python path:" -ForegroundColor Cyan
Write-Host "  uv pip install --python `"$selectedPython`" <package>" -ForegroundColor Green
Write-Host ""
Write-Host "[INFO] Or set UV_PYTHON environment variable:" -ForegroundColor Cyan
Write-Host "  `$env:UV_PYTHON = `"$selectedPython`"" -ForegroundColor Green
Write-Host "  uv pip install <package>" -ForegroundColor Green


































































































