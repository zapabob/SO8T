# 業界標準ベンチマークABCテスト実行とHFモデルカード更新

## 実装情報
- **日付**: 2026-01-24
- **Worktree**: main
- **機能名**: 業界標準ベンチマークABCテスト実行とHFモデルカード更新
- **実装者**: AI Agent

## 実装完了状況

✅ すべてのスクリプトは既に実装済みです。以下の手順で実行します。

## 実行手順

### 1. ABCテスト実行（10ランダムシード）

**スクリプト**: `scripts/evaluation/run_comprehensive_abc_benchmark.py`

**実行コマンド**:
```bash
py -3 scripts/evaluation/run_comprehensive_abc_benchmark.py --num_samples 100 --num_seeds 10
```

**評価内容**:
- 業界標準ベンチマーク: MMLU (5-shot), BBH, CommonsenseQA, OpenBookQA, SocialIQA, PIQA, Winogrande, BoolQ
- 高度ベンチマーク: DROP, StrategyQA
- 日本語ベンチマーク: ELIZA-100
- ABCモデル: A (Qwen2.5-7B), B (SO8T-trained), C (AEGIS-Phi3.5)

**出力**: `results/abc_testing/comprehensive_abc_results.json`

**注意**: 実行には数時間かかる可能性があります（10ランダムシード × 11ベンチマーク × 3モデル）

### 2. 統計的分析実行

**スクリプト**: `scripts/evaluation/statistical_abc_analysis.py`

**実行コマンド**:
```bash
py -3 scripts/evaluation/statistical_abc_analysis.py --results_file results/abc_testing/comprehensive_abc_results.json
```

**分析内容**:
- 多重比較補正（Bonferroni, FDR）
- エラーバー計算（95%信頼区間）
- 統計的有意性検定（t検定、Mann-Whitney U）
- 効果量計算（Cohen's d）

**出力**: `results/abc_testing/statistical_analysis.json`

### 3. グラフ可視化実行

**スクリプト**: `scripts/evaluation/visualize_abc_benchmark_statistics.py`

**実行コマンド**:
```bash
py -3 scripts/evaluation/visualize_abc_benchmark_statistics.py --results_file results/abc_testing/comprehensive_abc_results.json --statistical_analysis_file results/abc_testing/statistical_analysis.json
```

**生成グラフ**:
1. 個別ベンチマーク比較
2. 包括的ベンチマーク概要
3. 統計的有意性可視化
4. 業界標準比較
5. モデルランキングヒートマップ
6. **業界標準ベンチマーク分類グラフ**（MMLU含む、5-shotプロトコル）
7. **高度ベンチマーク分類グラフ**
8. **ELIZA-100分類グラフ**

**出力**: `results/abc_testing/visualizations/*.png`

### 4. HF README/モデルカード更新

**スクリプト**: `scripts/utils/update_hf_readme_with_benchmarks.py`

**実行コマンド**:
```bash
py -3 scripts/utils/update_hf_readme_with_benchmarks.py --repo_id <repo_id> --results_file results/abc_testing/comprehensive_abc_results.json --statistical_analysis_file results/abc_testing/statistical_analysis.json --visualization_dir results/abc_testing/visualizations
```

**更新内容**:
- ベンチマーク結果テーブル（エラーバー付きスコア）
- グラフ画像の埋め込み（業界標準/高度/ELIZA-100分類グラフ含む）
- 統計的有意性の説明
- MMLU 5-shotプロトコルの説明

## 自動実行スクリプト

### PowerShell版
```powershell
.\scripts\evaluation\run_abc_pipeline.ps1
```

### Bash版
```bash
bash scripts/evaluation/run_abc_pipeline.sh
```

### 待機実行スクリプト（ABCテスト完了を待って自動実行）
```bash
py -3 scripts/evaluation/wait_and_run_pipeline.py
```

## 実行状況確認

### 結果ファイルの確認
```powershell
Test-Path results/abc_testing/comprehensive_abc_results.json
Test-Path results/abc_testing/statistical_analysis.json
Get-ChildItem results/abc_testing/visualizations/*.png
```

### 実行ログの確認
```powershell
Get-Content results/abc_testing/*.log -Tail 50
```

## 実装の特徴

### 業界標準測定手法
- **MMLU**: 5-shot few-shot評価（業界標準プロトコル）
- **統計的堅牢性**: 10ランダムシードで評価
- **多重比較補正**: Bonferroni補正とFDR補正を実装
- **エラーバー表示**: 95%信頼区間を可視化
- **統計的有意性**: p値に基づく有意性マーカー（***, **, *）

### エラーバー付きグラフ
- 業界標準ベンチマーク分類グラフ（MMLU含む、5-shotプロトコル）
- 高度ベンチマーク分類グラフ（DROP, StrategyQA）
- ELIZA-100分類グラフ（日本語評価）

### HFモデルカードへの埋め込み
- ベンチマーク結果テーブル（エラーバー付きスコア表示）
- グラフ画像の埋め込み（業界標準/高度/ELIZA-100分類グラフ含む）
- 統計的手法の説明（多重比較補正、エラーバー計算、有意性検定）

## トラブルシューティング

### ABCテストがタイムアウトする場合
- `--num_samples`を減らす（例: 50）
- `--num_seeds`を減らす（例: 5）
- バックグラウンドで実行: `Start-Process py -ArgumentList "-3 scripts/evaluation/run_comprehensive_abc_benchmark.py --num_samples 100 --num_seeds 10"`

### メモリ不足の場合
- ベンチマークを個別に実行
- サンプル数を減らす

### モデル読み込みエラーの場合
- モデルパスを確認
- CUDA/GPUの利用可能性を確認

## 次のアクション

1. **ABCテスト実行**: バックグラウンドで実行中または実行待ち
2. **統計的分析実行**: ABCテスト完了後に実行
3. **グラフ可視化実行**: 統計的分析完了後に実行
4. **HF README更新**: すべてのグラフ生成後に実行（repo_idが必要）

## 実装ファイル一覧

- `scripts/evaluation/run_comprehensive_abc_benchmark.py`: ABCテスト実行スクリプト
- `scripts/evaluation/statistical_abc_analysis.py`: 統計的分析スクリプト
- `scripts/evaluation/visualize_abc_benchmark_statistics.py`: グラフ可視化スクリプト
- `scripts/utils/update_hf_readme_with_benchmarks.py`: HF README更新スクリプト
- `scripts/evaluation/run_abc_pipeline.ps1`: PowerShell自動実行スクリプト
- `scripts/evaluation/run_abc_pipeline.sh`: Bash自動実行スクリプト
- `scripts/evaluation/wait_and_run_pipeline.py`: 待機実行スクリプト
