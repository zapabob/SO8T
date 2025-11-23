# SO8T /think エンドポイント実装ログ

## 実装日時
2025-11-07

## 概要
SO8Tモデル用の`/think`エンドポイントを実装しました。内部推論プロセスを非公開で実行し、安全評価とVerifier評価を経て、要約した最終回答を返す本番運用可能なREST APIです。

## 実装ファイル
- `scripts/serve_think_api.py`: `/think`エンドポイントの実装

## 機能詳細

### 1. `/think`エンドポイントのフロー

1. **内部推論プロンプト構築**
   - ユーザークエリから内部推論用プロンプトを生成
   - `<think>...</think>`タグを使用して内部推論をマーク

2. **内部推論生成**
   - `SafetyAwareSO8TModel.base_model.generate()`を使用
   - 内部推論テキストを生成（外部に返さない）

3. **安全評価・Verifier評価**
   - `SafetyAwareSO8TModel.safety_gate()`で`safety_logits`を取得
   - `SafetyAwareSO8TModel.forward()`で`verifier_scores`を取得
   - 安全判定: `ALLOW`, `ESCALATE`, `REFUSE`
   - Verifier評価: 妥当性スコア（plausibility）と自己信頼度（self_confidence）

4. **最終回答生成**
   - `ALLOW`の場合: 内部推論を要約した最終回答を生成
   - `ESCALATE`/`REFUSE`の場合: 適切な拒否・エスカレーションメッセージを返す

5. **監査ログ**
   - `SQLMemoryManager`で監査ログを保存
   - ユーザーID、入力、内部推論ハッシュ、安全判定、Verifierスコアを記録

### 2. データモデル

#### ThinkRequest
- `user_id`: ユーザーID（必須、1-256文字）
- `query`: ユーザークエリ（必須、1-10000文字）
- `max_new_tokens`: 最大生成トークン数（デフォルト: 256、1-2048）
- `temperature`: サンプリング温度（デフォルト: 0.7、0.0-2.0）
- `top_p`: Top-pサンプリング（デフォルト: 0.9、0.0-1.0）

#### ThinkResponse
- `answer`: 最終回答
- `safety_label`: 安全判定ラベル（ALLOW/ESCALATE/REFUSE）
- `safety_conf`: 安全判定の信頼度（0.0-1.0）
- `verifier_plausibility`: Verifier妥当性スコア（オプション）
- `verifier_self_confidence`: Verifier自己信頼度（オプション）
- `escalated`: エスカレーションが必要かどうか
- `internal_reasoning_hash`: 内部推論のハッシュ値（監査用）

### 3. セキュリティ考慮事項

- **内部推論の非公開**: 内部推論テキストは外部に返さず、サービス内でのみ扱う
- **安全判定の優先**: 安全判定が`REFUSE`または`ESCALATE`の場合は即座に拒否
- **二重安全チェック**: 最終回答にも安全チェックを実行
- **監査ログ**: 内部推論ハッシュを記録（必要に応じて全文も記録可能）
- **入力検証**: Pydanticバリデーターで入力値を検証
- **SQLインジェクション対策**: ユーザーIDに危険な文字列を含めないよう検証

### 4. エラーハンドリング

- **モデルロードエラー**: モデルがロードできない場合、APIは起動するが`/think`エンドポイントはエラーを返す
- **データベースエラー**: データベースの初期化に失敗した場合、監査ログなしで続行
- **生成エラー**: テキスト生成に失敗した場合、適切なエラーメッセージを返す
- **評価エラー**: 安全評価・Verifier評価に失敗した場合、適切なエラーメッセージを返す

### 5. 監査ログ

#### イベントタイプ
- `think_internal_reasoning`: 内部推論の評価結果
- `think_escalation`: エスカレーション発生
- `think_final_answer_rejected`: 最終回答が安全チェックで拒否された
- `think_final_answer`: 正常な最終回答

#### 記録内容
- ユーザーID
- クエリ
- 内部推論ハッシュ
- 安全判定ラベルと信頼度
- Verifierスコア（妥当性、自己信頼度）
- タイムスタンプ

## 使用方法

### 環境変数
- `SO8T_BASE_MODEL`: ベースモデル名（デフォルト: `microsoft/Phi-3-mini-4k-instruct`）
- `SO8T_MODEL_PATH`: カスタムモデルパス（オプション）
- `SO8T_API_HOST`: APIホスト（デフォルト: `0.0.0.0`）
- `SO8T_API_PORT`: APIポート（デフォルト: `8000`）

### 起動方法
```bash
python scripts/serve_think_api.py
```

### APIリクエスト例
```bash
curl -X POST "http://localhost:8000/think" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user001",
    "query": "PythonでHello Worldを出力する方法を教えてください",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### APIレスポンス例
```json
{
  "answer": "PythonでHello Worldを出力するには、print()関数を使用します。\n\n例:\nprint('Hello World')\n\n実行すると、コンソールに「Hello World」が表示されます。",
  "safety_label": "ALLOW",
  "safety_conf": 0.95,
  "verifier_plausibility": 0.92,
  "verifier_self_confidence": 0.88,
  "escalated": false,
  "internal_reasoning_hash": "a1b2c3d4e5f6..."
}
```

## 統合ポイント

- `so8t-mmllm/src/models/safety_aware_so8t.py`: `SafetyAwareSO8TModel`を使用
- `safety_sql/sqlmm.py`: `SQLMemoryManager`で監査ログ保存
- `safety_sql/schema.sql`: データベーススキーマ

## 技術的詳細

### プロンプト構築
- 内部推論プロンプト: `<think>...</think>`タグを使用
- 最終回答プロンプト: 内部推論を要約した回答を生成

### 安全評価
- `SafetyAwareSO8TModel.safety_gate()`: 安全判定を実行
- 信頼度が低い場合（デフォルト: 0.7未満）は`ESCALATE`に変更

### Verifier評価
- `SafetyAwareSO8TModel.forward()`: Verifierスコアを取得
- 妥当性スコア（plausibility）と自己信頼度（self_confidence）を返す

## 今後の改善点

1. **内部推論全文の保存オプション**: 監査ログに内部推論全文を保存するオプションを追加
2. **レート制限**: APIリクエストのレート制限を追加
3. **認証・認可**: ユーザー認証・認可機能を追加
4. **メトリクス**: Prometheus等のメトリクス収集機能を追加
5. **キャッシュ**: 頻繁に使用されるクエリのキャッシュ機能を追加

## 完了ステータス

- [x] FastAPIアプリケーションと基本的な構造の実装
- [x] 内部推論プロンプト構築関数の実装
- [x] 最終回答プロンプト構築関数の実装
- [x] 安全評価・Verifier評価関数の実装
- [x] `/think`エンドポイントの実装
- [x] 監査ログ機能の統合
- [x] エラーハンドリングとバリデーションの追加

## 関連ファイル

- `scripts/serve_think_api.py`: メイン実装ファイル
- `so8t-mmllm/src/models/safety_aware_so8t.py`: SO8Tモデル実装
- `safety_sql/sqlmm.py`: 監査ログ管理
- `safety_sql/schema.sql`: データベーススキーマ





















