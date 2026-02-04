# 残りのファイルを分類するスクリプト

# 音声関連ファイルを utils/audio/ に移動
$audioFiles = @(
    "play_audio_notification.py",
    "play_completion_sound.ps1",
    "audio_test_fixed.bat",
    "enhanced_audio_test.bat",
    "simple_audio_test.bat",
    "test_audio.bat",
    "test_audio_now.bat",
    "volume_up.bat"
)

# 蒸留関連ファイルを training/distillation/ に移動
$distillationFiles = @(
    "distill_japanese_simple.py",
    "distill_phi31_to_japanese.py",
    "distill_so8t_knowledge.py",
    "distill_so8t_phi31_japanese_advanced.py",
    "simple_so8t_distillation.py",
    "run_so8t_distillation.py",
    "run_so8t_distillation.bat",
    "run_simple_distillation.bat"
)

# 可視化関連ファイルを evaluation/visualization/ に移動
$visualizationFiles = @(
    "visualize_ab_test_training_curves.py",
    "visualize_inference.py",
    "visualize_safety_training.py",
    "visualize_training.py",
    "create_paper_plots.py"
)

# データ生成関連ファイルを data/generation/ に移動
$dataGenerationFiles = @(
    "generate_synthetic_data.py",
    "generate_synthetic_japanese.py",
    "dataset_synth.py",
    "prepare_data.py"
)

# テスト関連ファイルを inference/tests/ に移動
$testFiles = @(
    "test_so8t_advanced_complex.bat",
    "test_so8t_complex.bat",
    "test_so8t_ollama_complex.bat",
    "test_so8t_ollama_complex_f16.bat",
    "test_so8t_qwen2vl.bat",
    "so8t_longtext_regression_test.py",
    "so8t_triality_ollama_test.py",
    "validate_so8t_gguf_model.py"
)

# Ollamaテスト関連ファイルを inference/ollama/ に移動
$ollamaTestFiles = @(
    "ollama_complex_test.ps1",
    "ollama_english_complex_test.bat",
    "ollama_extreme_complex_test.bat",
    "ollama_integration_test.bat",
    "ollama_qwen_so8t_lightweight_complex_test.bat",
    "ollama_simple_test.bat"
)

# 日本語言語処理関連ファイルを training/japanese/ に移動
$japaneseFiles = @(
    "enhanced_japanese_finetuning.py",
    "japanese_finetuning_script.py",
    "simple_japanese_finetuning.py",
    "japanese_enhanced_test.bat",
    "japanese_enhanced_v2_test.bat",
    "simple_japanese_test.bat"
)

# 実験・研究関連ファイルを training/experiments/ に移動
$experimentFiles = @(
    "pet_schedule_experiment.py",
    "temperature_calibration.py",
    "so8t_calibration.py",
    "weight_stability_demo.py"
)

# 実装関連ファイルを training/implementation/ に移動
$implementationFiles = @(
    "implement_reality_so8t.py",
    "implement_so8t_soul.py"
)

# その他のユーティリティファイル
$otherUtilsFiles = @(
    "build_vocab.py",
    "create_dummy_model.py",
    "create_gguf_file.py",
    "create_lightweight_gguf.py",
    "download_model.py",
    "export_gguf.py",
    "load_so8t_distilled_model.py",
    "organize_repository.py",
    "impl_logger.py",
    "so8t_conversion_logger.py"
)

# レポート生成関連ファイルを evaluation/reports/ に移動
$reportFiles = @(
    "final_research_summary.py",
    "generate_final_report.py",
    "generate_test_report.py",
    "summarize_results.py"
)

# 実行・デモ関連ファイルを inference/demos/ に移動
$demoFiles = @(
    "demonstrate_inference.py",
    "demonstrate_safety_inference.py",
    "run_complete_demo.bat",
    "run_so8t_external_demo.bat"
)

# パイプライン実行関連ファイル（まだ残っているもの）
$pipelineFiles = @(
    "run_ab_test_complete.py",
    "run_parallel_train_crawl.py",
    "run_safety_complete.py",
    "run_train_safety.py",
    "run_so8t_rtx3060_full_pipeline.bat",
    "run_pipeline_with_sound.bat"
)

# その他の実行スクリプト
$otherScripts = @(
    "infer.py",
    "inference.py",
    "train.py",
    "eval.py",
    "tasks.py",
    "monitor_training.py",
    "auto_resume.py",
    "auto_resume_startup.bat",
    "restore_epoch1_safety.py",
    "safety_losses.py",
    "so8t_burnin_qc.py",
    "mini_agi_loop.py",
    "run_safety.bat",
    "run_safety.ps1",
    "run_safety_pipeline.ps1",
    "run_comprehensive_tests.bat",
    "run_comprehensive_tests.ps1",
    "run_so8t_tests_automated.bat",
    "run_lightweight_test.bat",
    "run_weight_stability_demo.bat",
    "convert_qwen2vl_to_so8t_gguf.bat",
    "quantize_so8t_phi31_32gb.py",
    "quantize_so8t_phi31_lightweight.py",
    "_test_cuda.py"
)

# ディレクトリを作成
$directories = @(
    "scripts/utils/audio",
    "scripts/training/distillation",
    "scripts/evaluation/visualization",
    "scripts/data/generation",
    "scripts/inference/tests",
    "scripts/inference/ollama",
    "scripts/training/japanese",
    "scripts/training/experiments",
    "scripts/training/implementation",
    "scripts/evaluation/reports",
    "scripts/inference/demos"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# ファイルを移動する関数
function Move-Files {
    param(
        [string[]]$Files,
        [string]$Destination
    )
    
    foreach ($file in $Files) {
        $sourcePath = "scripts\$file"
        $destPath = "$Destination\$file"
        
        if (Test-Path $sourcePath) {
            Move-Item -Path $sourcePath -Destination $destPath -Force
            git add $destPath
            Write-Host "Moved $file to $Destination"
        }
    }
}

# ファイルを移動
Write-Host "`nMoving audio files..."
Move-Files -Files $audioFiles -Destination "scripts/utils/audio"

Write-Host "`nMoving distillation files..."
Move-Files -Files $distillationFiles -Destination "scripts/training/distillation"

Write-Host "`nMoving visualization files..."
Move-Files -Files $visualizationFiles -Destination "scripts/evaluation/visualization"

Write-Host "`nMoving data generation files..."
Move-Files -Files $dataGenerationFiles -Destination "scripts/data/generation"

Write-Host "`nMoving test files..."
Move-Files -Files $testFiles -Destination "scripts/inference/tests"

Write-Host "`nMoving Ollama test files..."
Move-Files -Files $ollamaTestFiles -Destination "scripts/inference/ollama"

Write-Host "`nMoving Japanese files..."
Move-Files -Files $japaneseFiles -Destination "scripts/training/japanese"

Write-Host "`nMoving experiment files..."
Move-Files -Files $experimentFiles -Destination "scripts/training/experiments"

Write-Host "`nMoving implementation files..."
Move-Files -Files $implementationFiles -Destination "scripts/training/implementation"

Write-Host "`nMoving report files..."
Move-Files -Files $reportFiles -Destination "scripts/evaluation/reports"

Write-Host "`nMoving demo files..."
Move-Files -Files $demoFiles -Destination "scripts/inference/demos"

Write-Host "`nMoving pipeline files..."
Move-Files -Files $pipelineFiles -Destination "scripts/pipelines"

Write-Host "`nMoving other utility files..."
Move-Files -Files $otherUtilsFiles -Destination "scripts/utils"

Write-Host "`nDone! Remaining files in scripts/ root:"
Get-ChildItem scripts\*.py, scripts\*.bat, scripts\*.ps1 | Where-Object { $_.DirectoryName -eq (Resolve-Path scripts).Path } | Select-Object Name

