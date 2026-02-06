# 2026-02-06 BF16 GGUF 変換ログ

## 概要

Model A (Microsoft Phi-3.5-mini-instruct) および Model B (AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp) の BF16 GGUF 形式への変換を完了しました。

## 変換結果

| モデル             | 入力パス                        | 出力パス                                       | ファイルサイズ |
| ------------------ | ------------------------------- | ---------------------------------------------- | -------------- |
| Model A (Baseline) | `H:\from_D\SO8T_models\model_a` | `H:\from_D\SO8T_models\gguf\model_a.bf16.gguf` | 7.64GB         |
| Model B (Borea)    | `H:\from_D\SO8T_models\model_b` | `H:\from_D\SO8T_models\gguf\model_b.bf16.gguf` | 7.64GB         |

## 使用ツール

- **変換スクリプト**: `src/infrastructure/external/llama.cpp-master/convert_hf_to_gguf.py`
- **出力形式**: BF16 (bfloat16)
- **テンソル数**: 197

## コマンド例

```powershell
py -3 src\infrastructure\external\llama.cpp-master\convert_hf_to_gguf.py H:\from_D\SO8T_models\model_a --outfile H:\from_D\SO8T_models\gguf\model_a.bf16.gguf --outtype bf16
py -3 src\infrastructure\external\llama.cpp-master\convert_hf_to_gguf.py H:\from_D\SO8T_models\model_b --outfile H:\from_D\SO8T_models\gguf\model_b.bf16.gguf --outtype bf16
```

## 次のステップ

- Model C (AEGIS-phi3.5-jp-v3.0): Enhanced Moonshot Pipeline (Phases 2-4) 完了後にベンチマーク実施
- A/B/C 統計ベンチマーク: ANOVA/Cohen's d 分析

## 関連ドキュメント

- `_docs/2026-02-06_AEGIS_v3_Meta_Prompt.md` - パイプライン引き継ぎメタプロンプト
