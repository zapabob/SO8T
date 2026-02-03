# AEGIS-phi3.5mini-jp-v3.0 統合計画書（最終版）

## 1. プロジェクト概要

### 1.1 基本情報
| 項目 | 内容 |
|------|------|
| プロジェクト名 | AEGIS-phi3.5mini-jp-v3.0 |
| バージョン | v2.5 → v3.0 アップグレー |
| ベースモデル | AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp |
| 目標アーキテクチャ | mHC + SO8T四重推論 |
| 量子化対応 | imatrix適用GGUF |
| 訓練方式 | SFT + GRPO (二段階) |

### 1.2 開発目標
AEGIS-v3.0 開発目標:
├─ mHC (Manifold-Constrained Hyper-Connections) の実装
├─ SO8T四重推論構造の完全統合
├─ ツールコーリング能力の強化
├─ 科学的推論能力の向上
└─ 量子化精度の保護（imatrix適用）

---

## 2. データソース統合（全8ソース）

### 2.1 データソース一覧
AEGIS-v3.0 データソース統合:
┌─────────────────────────────────────────────────────────────────────┐
│                      データ収集パイプライン                          │
└─────────────────────────────────────────────────────────────────────┘
│
        ┌─────────────────────┼─────────────────────┐
│                     │                     │
        ↓                     ↓                     ↓
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ ローカル      │   │ HF CLI        │   │ 外部API       │
│ データセット  │   │ データセット  │   │ 収集          │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        ├─ ArXiv引用データ    ├─ skill/MCP         ├─ WebResearch
        ├─ 防衛・JAXAデータ   ├─ DeepResearch      └─ リアルタイム検索
        ├─ NSFWデータ        └─ file operation    │
        └─ 四重推論データ                            │
                                ┌───────────────────┴───────────────────┐
↓                   ↓                   ↓
                        ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ HF Dataset A │   │ HF Dataset B │   │ HF Dataset C │
                        └───────────────┘   └───────────────┘   └───────────────┘

### 2.2 データソース詳細
| カテゴリ | ソース名 | アクセス方法 | サンプル数 | 用途 |
|---------|---------|-------------|-----------|------|
| 学術論文 | ArXiv 2024-2026 上位引用 | ローカルJSONL | 50,000 | 数学・科学推論 |
| 防衛データ | 日本の防衛関連 | ローカルJSONL | 10,000 | ドメイン特化 |
| 宇宙航空 | JAXA関連データ | ローカルJSONL | 10,000 | 物理推論 |
| 化学・薬物 | 薬物相互作用DB | ローカルJSONL | 10,000 | 化学推論 |
| 安全性 | NSFW・有害データ | ローカルJSONL | 15,000 | 安全拒否学習 |
| 推論データ | 四重推論<think>データ | ローカルJSONL | 25,000 | CoT学習 |
| ツール定義 | Skill/MCP関数定義 | HF CLI | 10,000 | ツール使用学習 |
| 検索・操作 | DeepResearch/WebSearch | HF CLI + API | 10,000 | 外部知識学習 |
| ファイル操作 | File Operationデータ | HF CLI | 10,000 | ファイル操作学習 |

### 2.3 HF CLIアクセス設定
`python
# scripts/data/hf_cli_collector.py
class HFDatasetCollector:
    """HuggingFace CLI経由データ収集"""
    
    def __init__(self):
        self.hf_cli_path = "/path/to/huggingface-cli"
    
    def download_dataset(self, dataset_name: str, save_dir: str):
        """データセットダウンロード"""
        cmd = [
            self.hf_cli_path,
            "datasets",
            "download",
            dataset_name,
            "--local-dir", save_dir,
            "--resume"
        ]
        subprocess.run(cmd)
    
    def list_available_datasets(self, filter_tags: List[str] = None):
        """利用可能なデータセット一覧"""
        cmd = [self.hf_cli_path, "datasets", "search"]
        if filter_tags:
            cmd.extend(["--filter", ",".join(filter_tags)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
`

### 2.4 ツールコーリングデータソース
ツールコーリング訓練データ:
┌─────────────────────────────────────────────────────────────────────┐
│                   ツール定義データ (skill/MCP)                       │
├─────────────────────────────────────────────────────────────────────┤
│ ├─ search: Web検索関数                                               │
│ ├─ file_read: ファイル読み込み                                        │
│ ├─ file_write: ファイル書き込み                                       │
│ ├─ python_exec: Python実行                                           │
│ ├─ data_analysis: データ解析                                         │
│ └─ calculator: 数式計算                                              │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                 検索・調査データ (DeepResearch)                       │
├─────────────────────────────────────────────────────────────────────┤
│ ├─ deep_research: 深層調査                                           │
│ ├─ web_search: Web検索                                              │
│ ├─ fact_check: ファクトチェック                                      │
│ └─ source_collect: ソース収集                                        │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                   ファイル操作データ                                  │
├─────────────────────────────────────────────────────────────────────┤
│ ├─ list_dir: ディレクトリ一覧                                        │
│ ├─ glob: ファイル glob                                              │
│ ├─ read_file: ファイル読み込み                                       │
│ ├─ write_file: ファイル書き込み                                      │
│ └─ edit_file: ファイル編集                                          │
└─────────────────────────────────────────────────────────────────────┘

---

## 3. データ処理パイプライン

### 3.1 統合データ処理フロー
データ処理パイプライン:
Raw Data Sources
      │
      ├─ Local JSONL Files
      ├─ HF CLI Downloads
      └─ API Collections
      │
      ↓
┌─────────────────────────────────────────────┐
│         データ統合・前処理                   │
│  ├─ フォーマット統一 (JSONL)                │
│  ├─ エンコーディング統一 (UTF-8)            │
│  ├─ 文字化け修正                            │
│  └─ 最小長フィルタリング                    │
└─────────────────────────────────────────────┘
      │
      ↓
┌─────────────────────────────────────────────┐
│         品質フィルタリング                   │
│  ├─ 重複除去 (SHA256ハッシュ)              │
│  ├─ 長さフィルタ (Instruction/Output)      │
│  ├─ 安全フィルタ (NSFW除去)                │
│  └─ 形式検証 (JSON形式)                    │
└─────────────────────────────────────────────┘
      │
      ↓
┌─────────────────────────────────────────────┐
│         データ変換・拡張                     │
│  ├─ SO8T四重推論チェーン追加               │
│  ├─ ツール定義形式変換                     │
│  ├─ 難易度分類                              │
│  └─ カテゴリタグ付け                        │
└─────────────────────────────────────────────┘
      │
      ↓
┌─────────────────────────────────────────────┐
│         最終データセット                     │
│  ├─ SFT用: 100,000サンプル                 │
│  ├─ GRPO用: 50,000サンプル                 │
│  └─ imatrix用: 31,857サンプル              │
└─────────────────────────────────────────────┘

### 3.2 四重推論データ変換
`python
# scripts/data/quad_thinking_converter.py
def convert_think_data_to_quad_format(
    input_file: str,
    output_file: str,
    min_length: int = 100
):
    """<think>データを四重推論形式に変換"""
    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as w:
        for line in f:
            data = json.loads(line)
            think_content = extract_think_content(data['text'])
            
            # 四重推論構造に分割
            quad_steps = parse_quad_inference(think_content)
            
            # 変換後のデータ
            converted = {
                'prompt': data.get('instruction', data.get('prompt', '')),
                'system_prompt': SO8T_SYSTEM_PROMPT,
                'quad_inference_chain': quad_steps,
                'answer': data.get('output', data.get('answer', '')),
                'source': 'think_conversion',
                'difficulty': classify_difficulty(quad_steps),
                'category': classify_category(data),
            }
            
            w.write(json.dumps(converted, ensure_ascii=False) + "\n")
`

### 3.3 HF CLIデータ収集スクリプト
`ash
#!/bin/bash
# scripts/data/download_hf_datasets.sh

# HF CLI設定
export HF_HOME="H:/from_D/webdataset/hf_cache"

# skill/MCP関連データセット
echo "[1/3] Downloading skill/MCP datasets..."
huggingface-cli download \
    --local-dir data/hf_datasets/skill_mcp \
    --resume \
    skill-collection-main \
    mcp-datasets \
    function-calling-benchmark

# DeepResearch/WebSearch関連
echo "[2/3] Downloading DeepResearch datasets..."
huggingface-cli download \
    --local-dir data/hf_datasets/deepresearch \
    --resume \
    deepseek-r1 \
    web-search-benchmark \
    research-question-dataset

# File Operation関連
echo "[3/3] Downloading File Operation datasets..."
huggingface-cli download \
    --local-dir data/hf_datasets/file_operation \
    --resume \
    file-operation-dataset \
    code-generation-benchmark

echo "Download complete!"
`

---

## 4. データ配分詳細

### 4.1 SFT訓練データ (100,000サンプル)
| ソース | アクセス | サンプル数 | 比率 | 重点項目 |
|--------|---------|-----------|------|---------|
| ArXiv 2024-2026 | Local | 25,000 | 25% | 数学・科学推論 |
| 四重推論<think> | Local | 20,000 | 20% | CoT学習 |
| Skill/MCP | HF CLI | 12,000 | 12% | ツール使用基礎 |
| DeepResearch | HF CLI | 10,000 | 10% | 調査能力 |
| File Operation | HF CLI | 8,000 | 8% | ファイル操作 |
| 防衛データ | Local | 8,000 | 8% | ドメイン特化 |
| JAXAデータ | Local | 7,000 | 7% | 物理推論 |
| 薬物データ | Local | 5,000 | 5% | 化学推論 |
| NSFW | Local | 5,000 | 5% | 安全拒否学習 |
| 合計 | - | 100,000 | 100% | - |

### 4.2 GRPO訓練データ (50,000サンプル)
| ソース | アクセス | サンプル数 | 比率 | 報酬重点 |
|--------|---------|-----------|------|---------|
| ArXiv計算問題 | Local | 15,000 | 30% | 数学正解報酬 |
| 四重推論 | Local + HF | 12,000 | 24% | CoT品質報酬 |
| ツール問い | HF CLI | 10,000 | 20% | ツール使用報酬 |
| 検索・調査 | HF CLI | 8,000 | 16% | 調査品質報酬 |
| NSFW | Local | 5,000 | 10% | 安全拒否報酬 |
| 合計 | - | 50,000 | 100% | - |

### 4.3 imatrixキャリブレーションデータ (31,857サンプル)
| ソース | アクセス | サンプル数 | 保護対象 |
|--------|---------|-----------|---------|
| ArXiv数学・物理 | Local | 18,000 | 数学計算 |
| JAXA物理計算 | Local | 5,000 | 物理推論 |
| 薬物構造 | Local + HF | 3,857 | 化学構造 |
| Skill/MCP計算 | HF CLI | 2,500 | ツール計算 |
| カスタム計算 | 生成 | 2,500 | 多様性 |
| **合計** | - | **31,857** | - |

---

# 5. 技術アーキテクチャ

## 5.1 SO8T + mHC統合アーキテクチャ
AEGIS-v3.0 モデルアーキテクチャ:
`
┌─────────────────────────────────────────────────────────────────────┐
│                        入力層                                        │
│  prompt + system_prompt + available_tools + quad_template           │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│             SO8T 四重推論ヘッド (4-Step CoT)                         │
│  Step1: Problem Formulation                                          │
│  Step2: Theoretical Approach                                         │
│  Step3: Computational Verification                                   │
│  Step4: Insight Conclusion                                           │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│           mHC Residual Manifold Layer (Sinkhorn)                     │
│  - H_res を双確率行列へ射影                                           │
│  - H_pre / H_post による読出・書込                                   │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│             Transformer Core (Base: Borea-Phi3.5)                    │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│               Tool-Calling Router + Safety Adapter                   │
└─────────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          Final Output                                │
└─────────────────────────────────────────────────────────────────────┘
`

## 5.2 mHC数式・Sinkhorn設定
### 5.2.1 mHC残差混合
mHCは残差混合行列 H_res を双確率行列に制約し、恒等性を保ちながらストリーム間混合を行う。

H_res ≥ 0, Σ_j H_res^{ij}=1, Σ_i H_res^{ij}=1

### 5.2.2 Sinkhorn-Knopp 正規化
H^{(k+1)} = T_row(T_col(H^{(k)}))

### 5.2.3 実装パラメータ（推奨）
| 項目 | 値 |
|------|----|
| Expansion Rate | B=4 |
| Sinkhorn Iter | 20 |
| Hidden Dim | 128 |
| Gate Init | 0.01 |

## 5.3 Tool-Calling Router 詳細
### 5.3.1 役割
- Tool使用の妥当性を推論チェーンに組み込む
- 不要なTool依存を弱報酬化

### 5.3.2 ルーティング構造
`
Quad-CoT Step2 → Tool-Need Prediction
     ↓
Tool Candidate Ranking
     ↓
Tool Call (if confidence > θ)
     ↓
結果を Step3/Step4 に反映
`

### 5.3.3 θしきい値
| モード | θ |
|-------|---|
| Conservative | 0.7 |
| Balanced | 0.5 |
| Aggressive | 0.35 |

---

# 6. 学習設計（SFT → GRPO）

## 6.1 SFTフェーズ
| 項目 | 内容 |
|------|------|
| 目的 | 基盤知識 + SO8T推論安定化 |
| データ | 100,000 |
| 量子化 | BF16 |
| バッチ | 2 |
| エポック | 3 |

## 6.2 GRPOフェーズ（報酬設計）
| ケース | 報酬 | 意図 |
|--------|------|------|
| 四重推論CoT正解 | +3.0 | 強化 |
| Tool正解 | +0.5 | 弱く推奨 |
| Tool誤答 | -2.0 | 強い罰 |
| Tool無し誤答 | -1.0 | 中罰 |

## 6.3 GRPO最終目的
- 正解は四重推論に依存する方が高報酬
- ツール正解は弱報酬
- ツール誤答は強罰
- 推論正解を最優先

---

# 7. 評価・統計解析（ANOVA / Tukey / 検出力）

## 7.1 評価ベンチ構成
- lm-evaluation-harness
- DeepEval
- HumanEval
- ABCテスト

## 7.2 評価対象モデル
| ID | モデル |
|----|--------|
| A | microsoft-phi3.5mini-instinct |
| B | AXCXEPT-Borea-phi3.5-instinct-jp |
| C | zapabobouj-AEGIS-phi3.5-jp-v3.0 |

## 7.3 統計分析プロトコル
| 手法 | 目的 | 出力 |
|------|------|------|
| ANOVA | 3モデル差異検定 | anova_summary.csv |
| Tukey HSD | 多重比較 | tukey_results.csv |
| 検出力解析 | p値の信頼性 | power_analysis.json |
| エラーバー | 平均±CI | errorbar.png |

---

# 8. GGUF + imatrix 出力

## 8.1 出力形式
| 種類 | 用途 |
|------|------|
| BF16 GGUF | 基準モデル |
| Q6_K GGUF | 推論高速 |
| Q5_K_M GGUF | バランス |
| Q4_K_M GGUF | 最大圧縮 |

## 8.2 imatrix適用効果
- 数学推論劣化を最小化
- SO8T構造保持率 >95%

---

# 9. HF Model Card / README 規約

## 9.1 必須記載
- 研究目的のみ / 商用禁止
- 引用文献・データ元
- 学習データ構成比率
- NSFW/防衛/薬物は拒否学習用途

---

# 10. 成果物ディレクトリ
`
outputs/
├── sft_model/
├── grpo_model/
├── gguf_models/
├── stats/
└── hf_card/
`

---

# 11. 最終まとめ
AEGIS-phi3.5mini-jp-v3.0 は mHC + SO8T四重推論 + Tool Calling + imatrix を統合した研究特化モデルであり、
- 四重推論で正解するほど強報酬
- ツール依存は弱報酬
- ツール誤答は強罰
- 推論正解を最優先
の報酬構造を実現する。
