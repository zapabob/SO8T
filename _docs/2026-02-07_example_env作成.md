# 2026-02-07_example_env作成

## 概要

プロジェクト設定のポータビリティ向上のため、`.env` および `env.env` から必要な変数を統合した `example.env` ファイルを作成しました。

## 変更内容

### 1. `example.env` の新規作成 [NEW]

- 以下のカテゴリで環境変数を整理:
  - Hugging Face 設定 (API Token)
  - AEGIS-v3.0 パイプライン設定
  - モデルおよびパス設定
  - AI および実行設定 (Ollama, Llama.cpp)
  - トレーニングハイパーパラメータ
  - GPU およびハードウェア設定
  - ロギングおよびデバッグ設定
  - オプションの機能フラグ (Unsloth 等の無効化)

## 使用方法

新たに環境を構築する際は、`example.env` を `.env` にコピーし、`HF_TOKEN` 等の機密情報を実際の値に書き換えて使用してください。

```powershell
copy example.env .env
```
