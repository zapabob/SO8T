# Phi3DecoderLayerでposition_embeddingsをself_attnに渡す修正 実装ログ

## 実装情報
- **日付**: 2025-11-15
- **Worktree**: main
- **機能名**: Phi3DecoderLayerでposition_embeddingsをself_attnに渡す修正
- **実装者**: AI Agent

## 実装内容

### 1. Phi3DecoderLayer.forwardでposition_embeddingsをself_attnに渡すように修正

**ファイル**: models/Borea-Phi-3.5-mini-Instruct-Jp/modeling_phi3.py

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-15  
**備考**: Phi3DecoderLayer.forwardがposition_embeddingsをkwargsで受け取っている場合、self.self_attn（Phi3Attention）に渡すように修正

- position_embeddingsをkwargsから取得
- position_embeddingsがNoneでない場合、self.self_attnに渡す
- position_embeddingsがNoneの場合は渡さない（内部で計算）

## 作成変更ファイル
- models/Borea-Phi-3.5-mini-Instruct-Jp/modeling_phi3.py

## 設計判断

1. **position_embeddingsの取得**: kwargs.get("position_embeddings", None)でposition_embeddingsを取得
2. **条件分岐**: position_embeddingsがNoneでない場合のみself.self_attnに渡す
3. **内部計算**: position_embeddingsがNoneの場合は渡さず、Phi3Attentionが内部で計算する

## テスト結果

- 学習プロセスを再起動し、エラーが解消されたことを確認
- TypeError: cannot unpack non-iterable NoneType objectエラーが発生しなくなった
- SO(8)群Transformerモデルが正常に初期化動作していることを確認（[SO8T] Initialized rotation gateと[SO8TPhi3Model] SO8T applied to 8/32 layersが表示）
- 学習が正常に進行していることを確認

## SO(8)群Transformerモデル確認

- **SO8TPhi3DecoderLayer**: position_idsを受け取り、SO8TPhi3Attentionに渡す（影響なし）
- **SO8TPhi3Attention**: position_idsを受け取り、内部でotary_embを使ってposition_embeddingsを計算（影響なし）
- **SO8T Rotation Gate**: 正常に初期化され、SO(8)群構造が維持されている

## 運用注意事項

### データ収集ポリシー
- 利用条件を守りつつ、高信頼ソースとして優先使用
- robots.txt遵守を徹底
- 個人情報機密情報の除外を徹底

### NSFWコーパス運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- モデル設計とドキュメントに明記
- 分類器は検出拒否用途のみ

### /thinkエンドポイント運用
- 四重Thinking部（<think-*>）は外部非公開を徹底
- <final>のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）
