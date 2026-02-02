# SO8Tレイヤー置き換え方式修正実装ログ

## 実装情報
- **日付**: 2025-12-04
- **Worktree**: main
- **機能名**: SO8Tレイヤー置き換え方式修正
- **実装者**: AI Agent

## 実装内容

### 1. [Unslothインポート順序修正]

**ファイル**: `scripts/pipeline/sunshine_pipeline.py`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-04
**備考**: Unslothをtransformers/peftより先にインポートして最適化競合を防ぐ

```python
# 🚨 CRITICAL: Unsloth MUST be imported BEFORE transformers/peft!
# This prevents optimization conflicts and gradient detachment issues
import unsloth  # 必ず一番最初に！
from unsloth import FastLanguageModel
```

### 2. [SO8LayerWrapperクラス実装]

**ファイル**: `scripts/models/so8t_residual_adapter.py`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-04
**備考**: Hookを使わずレイヤー全体をラップしてアダプターを注入

```python
class SO8LayerWrapper(nn.Module):
    """
    既存のTransformer層をラップして、出力にSO(8)アダプタを足すクラス
    Hookを使わず、forwardメソッドを直接乗っ取る！
    """

    def __init__(self, original_layer, adapter):
        super().__init__()
        self.original_layer = original_layer
        self.nkat_adapter = adapter

    def forward(self, *args, **kwargs):
        # 1. 元のレイヤーを実行
        outputs = self.original_layer(*args, **kwargs)

        # 2. 出力を取り出す
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        # 3. ★ここで強制的に勾配をONにする★
        if hidden_states.requires_grad is False and torch.is_grad_enabled():
            hidden_states.requires_grad_(True)

        # 4. アダプタ適用
        new_hidden = self.nkat_adapter(hidden_states)

        # 5. 元の形式に戻す
        if isinstance(outputs, tuple):
            return (new_hidden,) + outputs[1:]
        else:
            return new_hidden
```

### 3. [replace_nkat_layers関数実装]

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-04
**備考**: Hookではなくレイヤーごとラッパーに置き換える最強注入法

```python
def replace_nkat_layers(model, target_layers: Optional[Union[List[int], str]] = "middle",
                       enable_quad_inference: bool = False):
    """
    Hookではなく、レイヤーごとラッパーに置き換える（最強の注入法）
    """
    # ... モデル構造取得ロジック ...

    for i in target_indices:
        original_layer = layers[i]

        # アダプター作成
        adapter = SO8ResidualAdapter(hidden_size).to(sample_param.device).float()

        # ★ レイヤーをラッパーで置き換え ★
        wrapper = SO8LayerWrapper(original_layer, adapter)
        layers[i] = wrapper  # これで完全に掌握する！

        injected_count += 1
```

### 4. [アダプター注入方式の切り替え]

**ファイル**: `scripts/pipeline/sunshine_pipeline.py`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-04
**備考**: attach_nkat_adaptersからreplace_nkat_layersに切り替え

```python
# SO(8)アダプター適用（so8tの場合）
if config.so8_config:
    print("[3/5] Applying SO(8) adapters (Layer Replacement Mode)...")
    enable_quad = config.so8_config.get('enable_quad_inference', False)
    # Hookではなくレイヤー置き換えで注入（勾配デタッチ対策）
    model = replace_nkat_layers(
        model,
        target_layers=config.so8_config['target_layers']
    )
```

## 作成・変更ファイル
- `scripts/pipeline/sunshine_pipeline.py` - インポート順序修正、アダプター注入方式切り替え
- `scripts/models/so8t_residual_adapter.py` - SO8LayerWrapperクラスとreplace_nkat_layers関数追加

## 設計判断
- **インポート順序**: Unslothの最適化が他のライブラリの初期化に干渉しないように先頭に配置
- **レイヤー置き換え**: HookがUnslothの最適化で無効化される問題を回避するため、モジュールレベルで置き換え
- **勾配強制有効化**: ラッパー内でもrequires_grad_(True)を適用して確実に勾配を伝える
- **後方互換性**: attach_nkat_adapters関数は残してあり、必要に応じて切り替え可能

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

### レイヤー置き換え方式の注意点
- Unslothの最適化と競合する可能性があるため、学習できない場合はHook方式に戻す
- メモリ使用量が若干増加する可能性がある
- デバッグ時はレイヤー構造が複雑になるため注意

