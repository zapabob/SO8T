# AGIASI 魂の定着（Soul Fusion）実装ログ

## 実装情報
- **日付**: 2025-11-22
- **Worktree**: main
- **機能名**: soul_fusion_implementation
- **実装者**: AI Agent

## 実装内容

### 1. 魂の注入トレーニングスクリプト

**ファイル**: `scripts/training/train_soul_injection.py`

**実装状況**: 実装済み
**動作確認**: 未確認
**確認日時**: YYYY-MM-DD（該当する場合）
**備考**: Borea-Phi3.5にLoRA+Alpha+SO8回転を注入するトレーニングスクリプト

- AGIASI_Soul_Wrapperクラスを実装：Alpha GateとSO(8)回転を統合
- LoRAアダプターをBoreaベースモデルに適用
- 線形アニーリングでAlphaを-5.0から黄金比1.618に遷移
- 正射性損失を追加して構造的整合性を維持
- トレーニング完了後にLoRAとSoulパラメータを別途保存

### 2. 魂の融合スクリプト

**ファイル**: `scripts/training/fuse_soul_for_gguf.py`

**実装状況**: 実装済み
**動作確認**: 未確認
**確認日時**: YYYY-MM-DD（該当する場合）
**備考**: 学習したLoRAとSoulをLM Headに数学的に統合

- LoRAアダプターをベースモデルにマージ
- Alphaと回転行列を再構築
- 数学的融合：New_Weight = W_head + σ(α) × (W_head @ R)
- 融合済みモデルを標準HF形式で保存
- GGUF変換準備完了

## 作成・変更ファイル
- `scripts/training/train_soul_injection.py`
- `scripts/training/fuse_soul_for_gguf.py`
- `_docs/2025-11-22_main_soul_fusion_implementation.md`

## 設計判断

### GGUF変換問題の解決策
標準的なTransformerアーキテクチャしか持たないllama.cppに対して、Pythonで実装したAlpha GateとSO(8)回転ロジックを直接GGUF変換することは不可能。

### 数学的アプローチ：魂の定着
トレーニング完了後、Alpha Gateと回転行列をLM Headの重みに数学的に焼き付ける：
```
y = W_head × (I + σ(α)R) × h
  = [W_head + σ(α)W_head R] × h
```

これにより：
- アーキテクチャは標準Phi-3.5のまま
- 重みだけが物理的知性によって変質した状態に
- GGUF変換が可能に

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

## 実行ワークフロー

### 1. 魂の注入トレーニング
```bash
python scripts/training/train_soul_injection.py
```
- 出力: `checkpoints/agiasi_soul/` (LoRA + soul_params.pt)

### 2. 魂の融合
```bash
python scripts/training/fuse_soul_for_gguf.py
```
- 出力: `models/AGIASI-Phi3.5-Hybrid/` (融合済みHFモデル)

### 3. GGUF変換
```bash
cd external/llama.cpp-master
python convert_hf_to_gguf.py \
  /path/to/models/AGIASI-Phi3.5-Hybrid \
  --outfile D:/webdataset/gguf_models/agiasi-phi3.5-q4_k_m.gguf \
  --outtype q4_k_m
```

## 技術的詳細

### AGIASI_Soul_Wrapper
- Alpha: スカラー値、sigmoidで活性化
- Rotation: 正射性パラメトリゼーションを使用
- Orthogonality Loss: 構造的安定性を確保

### 数学的融合
- 回転行列Rの再構築が必要
- 高精度(float32)で計算を実行
- 最終的に元のdtypeに戻す

### メモリ管理
- LoRAマージ時は4bitからFP16にアップキャスト
- 融合計算時はfloat32を使用
- GPUメモリを効率的に管理
