# AEGIS命名規則更新実装ログ

## 実装情報
- **日付**: 2025-11-28
- **Worktree**: main
- **機能名**: AEGIS命名規則更新
- **実装者**: AI Agent

## 実装内容

### 1. ABCテストModel設定変更

**ファイル**: `configs/abc_test_config.json`, `scripts/evaluation/comprehensive_llm_benchmark.py`, `scripts/setup/setup_pipeline_environment.py`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-11-28
**備考**: Model AをGGUF BF16に、Model BをAEGIS-v2.0-Phi3.5-thinkingに変更

- **変更前**:
  - Model A: `borea_phi35_instruct_jp_q8_0.gguf` (GGUF Q8_0)
  - Model B: `borea_phi35_alpha_gate_sigmoid_bayesian/final` (AEGIS-Phi3.5-Enhanced)

- **変更後**:
  - Model A: `borea_phi35_instruct_jp_f16.gguf` (GGUF BF16)
  - Model B: `aegis_v2_phi35_thinking/final` (AEGIS-v2.0-Phi3.5-thinking)

### 2. HFアップロード設定変更

**ファイル**: `configs/hf_upload_config.yaml`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-11-28
**備考**: リポジトリ名とモデルカードを新しい命名規則に更新

- **変更内容**:
  - `base_repo_name`: "AEGIS-Phi3.5-Enhanced" → "AEGIS-v2.0-Phi3.5-thinking"
  - `title`: "AEGIS Phi-3.5 Enhanced" → "AEGIS v2.0 Phi-3.5 Thinking"
  - `tags`: "enhanced" → "thinking", "v2.0"追加

### 3. ドキュメントファイル一括変更

**ファイル**: `_docs/*.md`, `scripts/*.py`, `configs/*.yaml`, その他関連ファイル

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-11-28
**備考**: Pythonスクリプトで自動一括変更を実行

- **変更ルール**:
  - `AEGIS-Phi3.5-Enhanced` → `AEGIS-v2.0-Phi3.5-thinking`
  - `aegis-phi3.5-enhanced` → `aegis-v2.0-phi3.5-thinking`
  - `aegis_phi35_enhanced` → `aegis_v2_phi35_thinking`

- **変更結果**:
  - 処理ファイル数: 1,099個
  - 内容変更数: 69個
  - ファイル名変更数: 0個

## 作成・変更ファイル
- `configs/abc_test_config.json`
- `scripts/evaluation/comprehensive_llm_benchmark.py`
- `scripts/setup/setup_pipeline_environment.py`
- `configs/hf_upload_config.yaml`
- `scripts/rename_aegis_files.py` (新規作成)
- `_docs/*.md` (16個のドキュメントファイル)
- `scripts/*.py` (関連スクリプトファイル)
- `huggingface_upload/*` (アップロード関連ファイル)

## 設計判断
- **BF16選択理由**: Model Aとして最高精度のGGUF形式を使用するため
- **命名規則**: v2.0を明示的に入れ、thinkingを強調して進化を表現
- **一括変更**: 人的ミスを避けるためPythonスクリプトで自動化
- **後方互換性**: 古いファイル名は残さず完全移行

## 運用注意事項

### データ収集ポリシー
- 利用条件を守りつつ、高信頼ソースとして優先使用
- robots.txt遵守を徹底
- 個人情報・機密情報の除外を徹底

### NSFWコーパス運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- モデル設計とドキュメントに明記
- 分類器は検出・拒否用途のみ

### /thinkエンドポイント運用
- 四重Thinking部（`<think-*>`）は外部非公開を徹底
- `<final>`のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）
