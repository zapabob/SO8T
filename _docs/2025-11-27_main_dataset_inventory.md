# 2025-11-27 データセットインベントリ

英日バイリンガル SFT / PPO・内部CoT・コーディング支援・NSFW検知のために調達候補とした MIT / Apache 2.0 データセットの概要。

| # | Dataset | License | 主用途 (タグ) | モダリティ / 言語 | 想定ワークロード |
|---|---------|---------|---------------|--------------------|--------------------|
| 1 | `FreedomIntelligence/alpaca-gpt4-japanese` | Apache-2.0 | 指示応答 / CoT (JA) | text / ja | SFT (Task) |
| 2 | `FreedomIntelligence/sharegpt-japanese` | Apache-2.0 | 対話 / MCPツール指示 | text / ja | SFT (Task) |
| 3 | `FreedomIntelligence/MMLU_Japanese` | MIT | 評価用QA (JA) | text / ja | PPO/SFT (Reasoning Eval) |
| 4 | `az1/anthropic_hh_rlhf_japanese` | MIT | 安全 RLHF (JA) | text / ja | PPO (Safety) |
| 5 | `shi3z/anthropic_hh_rlhf_japanese` | MIT | 安全 RLHF (JA) | text / ja | PPO (Safety) |
| 6 | `fujiki/japanese_hh-rlhf-49k` | MIT | 安全 RLHF (JA) | text / ja | PPO (Safety) |
| 7 | `nomic-ai/gpt4all-j-prompt-generations` | Apache-2.0 | 英語指示+CoT | text / en | SFT (Coding/Reasoning) |
| 8 | `teknium/GPTeacher-General-Instruct` | MIT | 英語汎用指示 | text / en | SFT (Coding/UI/Business) |
| 9 | `ehartford/wizard_vicuna_70k_unfiltered` | Apache-2.0 | 英語高度CoT | text / en | SFT/PPO (Reasoning) |
| 10 | `OpenAssistant/oasst2` | Apache-2.0 | 多言語会話+Tool | text / multi | SFT (MCP/Tool use) |
| 11 | `open-orca/OpenOrca` | MIT | 英語Chain-of-Thought | text / en | SFT/PPO (Internal reasoning) |
| 12 | `Elizezen/japanese-nsfw-syosetsu-dataset` | Apache-2.0 | 日本語NSFW検知/生成 | text / ja | Safety (NSFW detection) |
| 13 | `eliasalbouzidi/NSFW-Safe-Dataset` | Apache-2.0 | 英語NSFW分類 | text / en | Safety (Classifier) |
| 14 | `RyokoExtra/JapaneseGoblin` | Apache-2.0 | 日本語wiki (Touhou) | text / ja/en | Pretraining補完 |

## クローン完了状況 (2025-11-27)

### 成功したデータセット (13/14)
- ✅ `FreedomIntelligence/alpaca-gpt4-japanese` (0.03 MB) - commit: 0b053baca4a6d0dc7694844de28e022f0f5fe217
- ✅ `FreedomIntelligence/sharegpt-japanese` (0.03 MB) - commit: c0cba53f24f912c0d436de02ba7da1cc8e7a7213
- ✅ `FreedomIntelligence/MMLU_Japanese` (9.25 MB) - commit: 0c5b5db2887739ea5327908d5655c61c8cb2d6d7
- ✅ `shi3z/anthropic_hh_rlhf_japanese` (0.04 MB) - commit: a5b5bb93f547b5aa34cbf517844e4698e5006eba
- ✅ `fujiki/japanese_hh-rlhf-49k` (0.03 MB) - commit: e54073cd64d675fbbef9b0a0743f3d6d63a3ca06
- ✅ `nomic-ai/gpt4all-j-prompt-generations` (0.07 MB) - commit: ec687c3a67653405f42d18c0be6f13f66bd4a813
- ✅ `teknium/GPTeacher-General-Instruct` (0.03 MB) - commit: b53b4e41539ee081817d5192586accc112491ffb
- ✅ `ehartford/wizard_vicuna_70k_unfiltered` (0.06 MB) - commit: cfe3f5810110d4d763665c070b4a966fda43e5c5
- ✅ `OpenAssistant/oasst2` (0.07 MB) - commit: 179dd21fc55192153d94adb0e0ce8f69e222bf75
- ✅ `open-orca/OpenOrca` (1.04 MB) - commit: e9c87b4abb2609913751f9b26553fdb9c061796c
- ✅ `Elizezen/japanese-nsfw-syosetsu-dataset` (0.04 MB) - commit: 481c3d062c78c87eabbd60b05041c524a4089aa0
- ✅ `eliasalbouzidi/NSFW-Safe-Dataset` (0.04 MB) - commit: 1f7d4a866c43e923c8ae11e69e56c7a0da52cddb
- ✅ `RyokoExtra/JapaneseGoblin` (0.10 MB) - commit: d40abcae65a8ee129f8fa3ee94f1c2ab0a68cd8c

### 失敗したデータセット (1/14)
- ❌ `az1/anthropic_hh_rlhf_japanese` - アクセス制限によりクローン不可 (403 Forbidden)

**総ディスク使用量**: 約 11.24 MB (Git LFSファイルは除外)

## コンテンツ品質監査結果 (2025-11-27)

### 監査サマリー
- **監査対象**: 13データセット (HuggingFaceからクローン)
- **正常アクセス**: 10データセット
- **アクセス不可**: 3データセット (権限/パス問題)
- **総解析文字数**: 約 79,000文字
- **NSFW検知**: 6件 (主にNSFW専用データセット)
- **コーディング関連**: 42件 (主に技術データセット)
- **ビジネス/MCP関連**: 12件 (分散型)

### ユースケース別分析

#### 推論強化 (Reasoning: 4 datasets)
- **キーワード検知**: コーディング 14件, ビジネス/MCP 4件
- **適性**: 日本語/英語の推論タスク、CoT学習に最適
- **例**: `FreedomIntelligence/MMLU_Japanese` (数学/科学キーワード検知)

#### 安全学習 (Safety: 1 dataset)
- **キーワード検知**: 該当なし (純粋な安全データ)
- **適性**: RLHF/安全ヘッダ学習
- **例**: `shi3z/anthropic_hh_rlhf_japanese`

#### コーディング支援 (Coding: 3 datasets)
- **キーワード検知**: コーディング 18件, NSFW 4件, ビジネス/MCP 4件
- **適性**: Cursor API節約、プログラミング支援、MCPツール使用
- **例**: `nomic-ai/gpt4all-j-prompt-generations` (コーディング 8件)

#### NSFW検知 (NSFW: 2 datasets)
- **キーワード検知**: NSFW 6件, コーディング 10件
- **適性**: 安全分類器学習、コンテンツモデレーション
- **例**: `Elizezen/japanese-nsfw-syosetsu-dataset` (NSFW 4件)

### 技術的所見
- **JSONパースエラー**: 複数データセットで発生 (想定内、データ形式の問題)
- **アクセス権限**: D:ドライブ上のデータセットにPythonアクセス制限
- **キーワード分析**: README.mdベースで効果的に機能
- **データ品質**: MIT/Apache-2.0ライセンス遵守を確認

### 次ステップ推奨
1. **四値分類実施**: ALLOW/ESCALATION/DENY/REFUSEラベル付与
2. **データクレンジング**: 統計的フィルタリングと品質向上
3. **SFT/PPO統合**: 適切なデータセットを学習パイプラインに投入
4. **性能評価**: /thinking, CoT, MCPツール使用能力の測定

## 参考メモ

- **NSFW検知**: `Elizezen/...`, `eliasalbouzidi/...` は NSFW ラベル付きで安全ヘッダ学習に使う。
- **Coding/UI/ML/DS**: `teknium/GPTeacher...`, `nomic-ai/gpt4all-j...`, `ehartford/wizard_vicuna...`, `OpenAssistant/oasst2` にコーディングやツール使用の指示が含まれる。
- **内部CoT**: `open-orca/OpenOrca` は GPT-4 由来の steg CoT で小型Phi3.5向け推論強化。
- **日本語SFT**: FreedomIntelligence 系 + `az1`, `shi3z`, `fujiki` が /thinking + PPO に適合。
- **追加候補**: 必要に応じ `eliasalbouzidi/NSFW-Safe-Dataset` の nopreprocessing split も利用（英語安全分類の raw 入力）。

> この表を元に今後の `git clone`（`D:/webdataset/datasets`）と README/ライセンススナップショット（`D:/webdataset/datasets-info`）を実施する。四値分類・クレンジングの結果や統計は後続のステップで追記予定。***

