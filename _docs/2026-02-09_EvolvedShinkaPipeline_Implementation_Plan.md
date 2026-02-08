# Evolved Shinka Pipeline 実装計画

## 実装ステータス

### 完了済み ✅

| ファイル | パス | 機能 |
|---------|------|------|
| 忘却曲線冻结 | `src/optimization/ebbinghaus_freeze.py` | R=exp(-t/S)ベースの動的冻结、imatrix連携 |
| 世界情勢データ | `src/data/world_events_2024_2026.py` | 28イベント（米ベネズエラ、日中、AI、科学数学等） |
| LLM Judge 95% | `src/evaluation/llm_judge_95.py` | Z-score 1.96で外れ値除去 |
| ShinkaNEAT統合 | `src/evolution/shinka_neat_engine.py` | NEAT+ShinkaEvolve+Ollama統合 |
| 四重推論生成 | `src/data/evolutionary/quadruple_vssi_generator.py` | VSSI形式データ生成 |
| 統合パイプライン | `src/infrastructure/pipeline/evolved_shinka_pipeline.py` | 全フェーズorchestrator |
| PowerShell自動再開 | `scripts/pipeline/power_on_auto_resume.ps1` | 電源投入時自動再開 |
| Task Scheduler登録 | `scripts/pipeline/install_scheduler.bat` | 管理者が管理者として実行 |
| テスト | `tests/test_evolved_pipeline_*.py` | インポート・機能テスト |

### 保留中 ⏳

## Phase 1: 統合とテスト（優先度高）

### 1.1 モジュールインポート問題修正

**問題**: `from src.evolution import ...` が失敗

**原因**: `src/` ディレクトリがパッケージとして認識されていない

**解決策**:
```python
# evolved_shinka_pipeline.py の修正
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "src"
for p in [project_root, src_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
```

### 1.2 Ollama統合テスト

**必要**: Borea-Phi-3.5-Instinct-JP が Ollama で動作確認

```powershell
ollama list
# borea-phi-3.5-instinct-jp が表示されるか確認

ollama run borea-phi-3.5-instinct-jp "你好、世界"
# 動作確認
```

### 1.3 統合テスト実行

```bash
py -3 tests/test_evolved_pipeline_imports.py
py -3 tests/test_evolved_pipeline_functional.py
```

## Phase 2: 機能拡張（優先度中）

### 2.1 ArXiv/BioRxiv統合

**目的**: 科学的論文データを世界情勢データに連携

```python
# src/data/world_events_2024_2026.py の拡張
class WorldEvents2024_2026:
    def integrate_arxiv_papers(self, papers: List[Dict]) -> None:
        """ArXiv論文を世界情勢に統合"""
        for paper in papers:
            event = WorldEvent(
                event_id=f"ARXIV_{paper['id']}",
                title=paper['title'],
                description=paper['summary'],
                category="science_math",
                scientific_relevance=True,
                related_papers=[f"arXiv:{paper['id']}"]
            )
            self.events[event.event_id] = event
```

### 2.2 ドメイン知識保護の強化

**目的**: ArXiv/BioRxiv/ドメイン知識を常に冻结

```python
# ebbinghaus_freeze.py の拡張
class EbbinghausFreeze:
    def __init__(self, ...):
        # 保護ドメインの初期知識を追加
        protected_domains = [
            "arxiv", "biorxiv", "domain_knowledge",
            "world_events_2024_2026", "science", "math",
            "quadruple_reasoning", "vssi"
        ]

        for domain in protected_domains:
            self.add_memory(
                content=f"Protected knowledge domain: {domain}",
                domain=domain,
                importance_score=0.95,
                strength_hours=168.0  # 1週間
            )
```

### 2.3 CoT Thinking形式対応

**目的**: 四重推論をCoT Thinking形式でも出力

```python
# quadruple_vssi_generator.py の拡張
class QuadrupleVSSIGenerator:
    def generate_cot_format(self, topic: str) -> Dict[str, str]:
        """CoT Thinking形式で出力"""
        quadruple = self.ollama_gen.generate_complete(topic)

        return {
            "instruction": quadruple.instruction,
            "cot_thinking": f"""
<think>
{quadruple.quadruple_reasoning.think_task}
---
{quadruple.quadruple_reasoning.think_analysis}
---
{quadruple.quadruple_reasoning.think_safety}
---
{quadruple.quadruple_reasoning.think_policy}
</think>
""".strip(),
            "output": quadruple.final_output
        }
```

## Phase 3: パフォーマンス最適化（優先度低）

### 3.1 並列処理

**目的**: 複数トピックを並列処理

```python
from concurrent.futures import ThreadPoolExecutor

def generate_parallel(self, topics: List[str], max_workers: int = 4) -> List[VSSIDataSample]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(self.generate_complete, topics))
    return results
```

### 3.2 キャッシュ機能

**目的**: 同じトピックを再利用

```python
class CachedQuadrupleGenerator:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cached(self, topic: str) -> Optional[VSSIDataSample]:
        cache_path = self.cache_dir / f"{hash(topic)}.json"
        if cache_path.exists():
            return VSSIDataSample(**json.loads(cache_path.read_text()))
        return None
```

## Phase 4: 監視・運用（優先度低）

### 4.1 Prometheusメトリクス

```python
from prometheus_client import Counter, Histogram, start_http_server

PIPELINE_RUNS = Counter('pipeline_runs_total', 'Total pipeline runs')
PIPELINE_DURATION = Histogram('pipeline_duration_seconds', 'Pipeline duration')

@PIPELINE_DURATION.time()
def run_pipeline():
    ...
```

### 4.2 アラート機能

```python
def send_alert(message: str, level: str = "WARNING"):
    """Slack/Teamsにアラート送信"""
    if level == "ERROR":
        # エラー時は停止
        pass
```

## 実行コマンド

### 標準実行

```powershell
.\scripts\pipeline\power_on_auto_resume.ps1
```

### チェックポイントから再開

```powershell
.\scripts\pipeline\power_on_auto_resume.ps1 -Resume
```

### Task Scheduler登録（管理者として実行）

```powershell
.\scripts\pipeline\install_scheduler.bat
```

### Python直接実行

```powershell
py -3 src/infrastructure/pipeline/evolved_shinka_pipeline.py --resume
```

### スキップオプション

```powershell
$env:SO8T_SKIP_EVOLUTION = "1"
$env:SO8T_SKIP_QUADRUPLE = "1"
$env:SO8T_SKIP_JUDGE = "1"
$env:SO8T_SKIP_CLEANSING = "1"
```

## リスクと対策

| リスク | 影響 | 対策 |
|-------|------|------|
| Ollama未起動 | パイプライン停止 | ヘルスチェック+再試行 |
| D:ドライブ未接続 | チェックポイント失敗 | フォールバックパス |
| メモリ不足 | 処理失敗 | バッチサイズ削減 |
| ネットワークエラー | データ収集失敗 | リトライ+チェックポイント |

## 成功指標

1. **インポートテスト**: 6/6モジュール正常インポート
2. **機能テスト**: 全テストパス
3. **実行時間**: 1トピックあたり5分以内
4. **チェックポイント**: 5分間隔で正常保存
5. **再開機能**: 電源断から30秒以内に再開

---

**作成日**: 2026-02-09
**最終更新**: 2026-02-09
**バージョン**: 1.0
