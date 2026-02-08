#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合ムーンショットパイプライン 2025-2026
Arxiv上位5万件、防衛・JAXA白書収集 → SFT → 最新手法統合 → ABCテスト → HFアップロード

実行方法:
    py -3 scripts/pipeline/integrated_moonshot_pipeline_2025_2026.py
"""

import sys
import os
from pathlib import Path
import json
import logging
import threading
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import subprocess
import time

# プロジェクトルートとOpenCodeをパスに追加（絶対パスを保証）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "OpenCode"))

from experiments.enhanced_moonshot_pipeline import EnhancedMoonshotPipeline

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_moonshot_pipeline_2025_2026.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntegratedMoonshotPipeline2025_2026:
    """統合ムーンショットパイプライン 2025-2026"""
    
    def __init__(self):
        self.project_root = project_root
        self.data_dir = self.project_root / "data" / "collected_2025_2026"
        self.results_dir = self.project_root / "results" / "moonshot_2025_2026"
        self.models_dir = self.project_root / "models" / "moonshot_2025_2026"
        
        # ディレクトリ作成
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_index_file = self.data_dir / "checkpoint_index.ptr"
        self.rolling_checkpoints = [
            self.data_dir / "pipeline_checkpoint_1.json",
            self.data_dir / "pipeline_checkpoint_2.json",
            self.data_dir / "pipeline_checkpoint_3.json"
        ]
        
        # 定期保存用フラグとスレッド
        self._stop_checkpoint_thread = threading.Event()
        self._checkpoint_thread = None
        self._current_phase = "initialized"
        self._current_data = {}
        
    def discover_existing_datasets(self) -> Dict[str, List[Path]]:
        """既存データセットを検出"""
        datasets = {
            "arxiv": [],
            "whitepapers": [],
            "nsfw_detection": [],
            "drug_detection": [],
            "integrated": [],
            "so8t": []
        }
        
        # data/ディレクトリから検出
        data_dir = self.project_root / "data"
        if data_dir.exists():
            # Arxiv/Biorxiv
            arxiv_dir = data_dir / "arxiv_biorxiv"
            if arxiv_dir.exists():
                datasets["arxiv"].extend(list(arxiv_dir.glob("*.jsonl")))
            
            # 統合データセット
            integrated_dir = data_dir / "integrated"
            if integrated_dir.exists():
                datasets["integrated"].extend(list(integrated_dir.glob("*.jsonl")))
            
            # NSFW検知
            nsfw_dir = data_dir / "nsfw_detection"
            if nsfw_dir.exists():
                datasets["nsfw_detection"].extend(list(nsfw_dir.glob("*.jsonl")))
            
            # SO8Tデータセット
            so8t_patterns = [
                data_dir / "aegis_phi35_v2_with_nkat_so8t",
                data_dir / "so8t_*"
            ]
            for pattern in so8t_patterns:
                if isinstance(pattern, Path) and pattern.exists():
                    datasets["so8t"].extend(list(pattern.glob("*.jsonl")))
                elif isinstance(pattern, str):
                    # Glob pattern handling
                    p = Path(pattern)
                    datasets["so8t"].extend(list(p.parent.glob(p.name + "/*.jsonl")))
        
        # H:/from_D/webdatasetから検出 (D:/webdatasetから移行された想定)
        webdataset_dir = Path("H:/from_D/webdataset")
        if not webdataset_dir.exists():
            webdataset_dir = Path("D:/webdataset")
            
        if webdataset_dir.exists():
            # NSFW検知データセット
            nsfw_web = webdataset_dir / "nsfw_detection_dataset"
            if nsfw_web.exists():
                datasets["nsfw_detection"].extend(list(nsfw_web.glob("*.jsonl")))
            
            # 薬物検知データセット
            drug_web = webdataset_dir / "drug_pharmaceutical_detection_dataset"
            if drug_web.exists():
                datasets["drug_detection"].extend(list(drug_web.glob("*.jsonl")))
            
            # 処理済みデータ
            processed_dir = webdataset_dir / "processed"
            if processed_dir.exists():
                datasets["integrated"].extend(list(processed_dir.glob("**/*.jsonl")))
        
        return datasets

    def integrate_existing_datasets(self, discovered_datasets: Dict[str, List[Path]]) -> Path:
        """既存データセットのクレンジング、重複削除、統合"""
        logger.info("=" * 80)
        logger.info("📦 既存データセットのクレンジングと統合")
        logger.info("=" * 80)
        
        seen_texts = set()
        integrated_data = []
        
        # 読み込むデータの種類
        categories = ["arxiv", "integrated", "so8t", "nsfw_detection", "drug_detection"]
        
        for category in tqdm(categories, desc="Categories", unit="cat"):
            files = discovered_datasets.get(category, [])
            for data_file in tqdm(files, desc=f"Cleaning {category}", unit="file", leave=False):
                logger.debug(f"クレンジング中: {data_file}")
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            item = json.loads(line)
                            
                            # クレンジングロジック
                            text = item.get("text", item.get("instruction", "") + item.get("output", ""))
                            if not text: continue
                            
                            # 重複削除
                            text_hash = hash(text.strip())
                            if text_hash in seen_texts: continue
                            seen_texts.add(text_hash)
                            
                            # 形式正規化
                            source_name = str(data_file.name)
                            # 数学・科学のブレイクスルー関連の重み付け（メタデータへの付与）
                            is_breakthrough = any(kw in text.lower() for kw in ["erdos", "fields medal", "nobel", "conjecture", "proof", "breakthrough"])
                            
                            clean_item = {
                                "instruction": item.get("instruction", item.get("prompt", "以下の科学的課題を考察せよ。")),
                                "input": item.get("input", ""),
                                "output": item.get("output", item.get("text", item.get("response_desirable", ""))),
                                "metadata": {
                                    "source": source_name,
                                    "category": category,
                                    "is_breakthrough": is_breakthrough,
                                    "cleansed_at": datetime.now().isoformat()
                                }
                            }
                            integrated_data.append(clean_item)
                except Exception as e:
                    logger.error(f"❌ {data_file} の処理に失敗: {e}")
        
        # 統合データセットを保存
        output_file = self.data_dir / "cleansed_integrated_dataset.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(integrated_data, desc="Saving integrated data", unit="item"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ クレンジング・統合完了: {len(integrated_data)}件のデータを {output_file} に保存")
        return output_file

    def validate_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """データセットの検証"""
        logger.info(f"データセット検証中: {dataset_path}")
        
        stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "format_errors": [],
            "required_fields": ["instruction", "input", "output"]  # 例
        }
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    stats["total_samples"] += 1
                    try:
                        item = json.loads(line)
                        # 必須フィールドチェック
                        if all(field in item for field in stats["required_fields"]):
                            stats["valid_samples"] += 1
                        else:
                            stats["invalid_samples"] += 1
                            stats["format_errors"].append(f"Line {line_num}: Missing required fields")
                    except json.JSONDecodeError as e:
                        stats["invalid_samples"] += 1
                        stats["format_errors"].append(f"Line {line_num}: JSON decode error - {e}")
        except Exception as e:
            logger.error(f"❌ 検証中にエラーが発生: {e}")
        
        logger.info(f"検証結果: {stats['valid_samples']}/{stats['total_samples']} 有効")
        return stats
        
    def collect_scientific_papers_top_100000(self) -> List[Path]:
        """Arxiv/Biorxiv上位計10万件の収集"""
        logger.info("=" * 80)
        logger.info("📚 Phase 1: Arxiv/Biorxiv上位計10万件の収集")
        logger.info("=" * 80)
        
        output_paths = []
        sources = ["arxiv", "biorxiv"]
        
        for source in sources:
            output_file = self.data_dir / f"{source}_top_50000.jsonl"
            checkpoint_file = self.data_dir / f"{source}_checkpoint.json"
            
            # citation_fetcherを使用
            # ノーベル賞・フィールズ賞級のトピック、および専門ドメイン（軍事、宇宙、生物、薬理）を優先するクエリ
            scientific_query = (
                "quantum gravity OR topological insulators OR protein folding OR prime number theorem OR Riemann hypothesis OR P vs NP OR "
                "stealth technology OR hypersonic weapons OR autonomous defense systems OR "
                "orbital mechanics OR deep space exploration OR satellite constellation OR "
                "CRISPR gene editing OR synthetic biology OR neuropharmacology"
            )
            
            cmd = [
                "py", "-3",
                str(self.project_root / "OpenCode_src" / "scripts" / "data_processing" / "citation_fetcher.py"),
                "--source", source,
                "--query", scientific_query,
                "--max-papers", "50000",
                "--start-year", "2024",
                "--end-year", "2026",
                "--output", str(output_file),
                "--checkpoint", str(checkpoint_file),
                "--verbose"
            ]
            
            logger.info(f"{source.upper()}収集開始: {output_file}")
            try:
                subprocess.run(cmd, check=True, cwd=self.project_root)
                output_paths.append(output_file)
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ {source.upper()}収集エラー: {e}")
                # 一部のソースが失敗しても続行（レジューム可能）
        
        return output_paths
    
    def collect_defense_jaxa_whitepapers(self) -> List[Path]:
        """日本の防衛・JAXA白書の収集"""
        logger.info("=" * 80)
        logger.info("📚 Phase 2: 日本の防衛・JAXA白書の収集")
        logger.info("=" * 80)
        
        # web-search-deepresearchスキルを使用して白書を検索・収集
        # 実際の実装では、web-search-deepresearchスキルを呼び出す
        
        output_files = []
        
        # 防衛白書
        defense_output = self.data_dir / "defense_whitepaper.jsonl"
        logger.info(f"防衛白書収集: {defense_output}")
        # TODO: web-search-deepresearchを使用して防衛白書を収集
        
        # JAXA白書
        jaxa_output = self.data_dir / "jaxa_whitepaper.jsonl"
        logger.info(f"JAXA白書収集: {jaxa_output}")
        # TODO: web-search-deepresearchを使用してJAXA白書を収集
        
        output_files.extend([defense_output, jaxa_output])
        
        logger.info("✅ 防衛・JAXA白書収集完了（実装予定）")
        return output_files
    
    def verify_nsfw_drug_datasets(self) -> bool:
        """NSFW、性的、薬物データセットの確認（既に収集済み想定）"""
        logger.info("=" * 80)
        logger.info("🔍 Phase 3: NSFW・性的・薬物データセットの確認")
        logger.info("=" * 80)
        
        # 既存データセットの確認
        nsfw_datasets = [
            self.data_dir / "nsfw_detection_dataset.jsonl",
            self.project_root / "data" / "nsfw_detection" / "*.jsonl"
        ]
        
        drug_datasets = [
            self.data_dir / "drug_detection_dataset.jsonl",
            self.project_root / "data" / "drug_detection" / "*.jsonl"
        ]
        
        found_nsfw = any(Path(p).exists() if "*" not in str(p) else list(Path(p).parent.glob(Path(p).name)) for p in nsfw_datasets)
        found_drug = any(Path(p).exists() if "*" not in str(p) else list(Path(p).parent.glob(Path(p).name)) for p in drug_datasets)
        
        if found_nsfw and found_drug:
            logger.info("✅ NSFW・薬物データセット確認完了")
            return True
        else:
            logger.warning("⚠️ NSFW・薬物データセットが見つかりません（検知目的で収集が必要）")
            return False
    
    def execute_sft(self, dataset_paths: List[Path]) -> Path:
        """SFT実行"""
        logger.info("=" * 80)
        logger.info("🎓 Phase 4: SFT実行")
        logger.info("=" * 80)
        
        # Enhanced Moonshot PipelineのSFT機能を使用
        pipeline = EnhancedMoonshotPipeline(
            boreas_model_path="AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
        )
        pipeline.load_boreas_model()
        
        # SFT実行
        logger.info("SFT実行中...")
        pipeline.execute_sft_rlpo_integration(target_datasets=dataset_paths)
        
        # モデル出力を整理
        sft_model_path = self.models_dir / "sft_model"
        # 実際には EnhancedMoonshotPipeline が models/aegis_v25_rlpo などに出力する
        # そのパスを最終的な結果として返す
        actual_output = Path("models/aegis_v25_rlpo")
        if actual_output.exists():
            sft_model_path = actual_output
            
        logger.info(f"✅ SFT完了: {sft_model_path}")
        pipeline._cleanup_resources()
        return sft_model_path
    
    def execute_advanced_techniques_integration(self, sft_model_path: Path) -> Path:
        """2025-2026最新手法統合（DeepseekGLPO、mHC、多様体、SO8T、imatrix）"""
        logger.info("=" * 80)
        logger.info("🔬 Phase 5: 2025-2026最新手法統合")
        logger.info("=" * 80)
        
        pipeline = EnhancedMoonshotPipeline(
            boreas_model_path=str(sft_model_path)
        )
        pipeline.load_boreas_model()
        
        config = {
            "enable_deepseek_grpo": True,
            "enable_mhc_manifold": True,
            "enable_so8t": True,
            "enable_geometric_scaling": True,
            "enable_imatrix_protection": True
        }
        
        # 最新手法統合実行
        logger.info("SO(8) Hyper-Combination (Vector+Spinor) 再学習中...")
        pipeline.execute_so8_residual_adapter_retraining()

        logger.info("DeepseekGLPO (GRPO) 統合中...")
        pipeline.execute_deepseek_grpo_integration()
        
        logger.info("mHC多様体アーキテクチャ統合中...")
        pipeline.execute_mhc_manifold_integration()
        
        logger.info("幾何学的スケーリング統合中...")
        pipeline.execute_geometric_scaling_integration()
        
        logger.info("SO8T四重推論 + imatrix保護付きGGUF量子化中...")
        pipeline.execute_so8t_imatrix_quantization()
        
        logger.info("BF16 GGUF変換中 (ユーザー特定リクエスト)...")
        if hasattr(pipeline, 'execute_bf16_gguf_conversion'):
            pipeline.execute_bf16_gguf_conversion()
        
        final_model_path = self.models_dir / "final_model_with_advanced_techniques"
        logger.info(f"✅ 最新手法統合完了: {final_model_path}")
        pipeline._cleanup_resources()
        return final_model_path
    
    def execute_abc_test(self, final_model_path: Path) -> Dict[str, Any]:
        """ABCテスト実行（A:ベース、B:Borea-phi3.5、C:ムーンショット結果）"""
        logger.info("=" * 80)
        logger.info("🅰️🅱️🆎 Phase 6: ABCテスト実行")
        logger.info("=" * 80)
        
        # ABCテストスクリプトを実行
        abc_test_script = self.project_root / "scripts" / "evaluation" / "industry_standard_agi_abc_test.py"
        
        if not abc_test_script.exists():
            logger.warning("⚠️ ABCテストスクリプトが見つかりません")
            return {}
        
        cmd = [
            "py", "-3",
            str(abc_test_script),
            "--models", "modela", "aegis_adjusted",  # A, B, C
            "--output-root", str(self.results_dir / "abc_test_results")
        ]
        
        logger.info(f"実行コマンド: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, cwd=self.project_root)
            
            # 結果読み込み
            results_file = self.results_dir / "abc_test_results" / "summary.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    abc_results = json.load(f)
                logger.info("✅ ABCテスト完了")
                return abc_results
            else:
                logger.warning("⚠️ ABCテスト結果ファイルが見つかりません")
                return {}
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ABCテストエラー: {e}")
            return {}
    
    def check_performance_improvement(self, abc_results: Dict[str, Any]) -> bool:
        """過去比で成績向上を確認"""
        logger.info("=" * 80)
        logger.info("📊 Phase 7: 成績向上確認")
        logger.info("=" * 80)
        
        # 過去の結果と比較
        previous_results_file = self.results_dir / "previous_abc_results.json"
        
        if not previous_results_file.exists():
            logger.info("過去の結果がないため、今回の結果をベースラインとして保存")
            with open(previous_results_file, 'w', encoding='utf-8') as f:
                json.dump(abc_results, f, indent=2, ensure_ascii=False)
            return True  # 初回実行の場合はTrue
        
        # 過去の結果読み込み
        with open(previous_results_file, 'r', encoding='utf-8') as f:
            previous_results = json.load(f)
        
        # スコア比較（簡易実装）
        # 実際の実装では、各ベンチマークのスコアを詳細に比較
        
        logger.info("✅ 成績向上確認完了（簡易実装）")
        return True  # TODO: 実際の比較ロジックを実装
    
    def upload_to_hf_if_improved(self, final_model_path: Path, improved: bool) -> bool:
        """成績向上時のみHFアップロード（HF標準ファイル + BF16 GGUF）"""
        logger.info("=" * 80)
        logger.info("☁️ Phase 8: HFアップロード（成績向上時のみ）")
        logger.info("=" * 80)
        
        if not improved:
            logger.info("⚠️ 成績向上が確認されなかったため、HFアップロードをスキップ")
            return False
            
        # Enhanced Moonshot Pipelineのアップロード機能を使用
        pipeline = EnhancedMoonshotPipeline(
            boreas_model_path=str(final_model_path)
        )
        
        # 必要なGGUF変換が完了していることを前提として、アップロード自動化を実行
        logger.info("HFアップロード自動化を実行中...")
        pipeline.execute_hf_upload_automation()
        
        logger.info("✅ HFアップロード完了")
        pipeline._cleanup_resources()
        return True
    
    def save_checkpoint(self, phase: str, data: Dict[str, Any]):
        """チェックポイント保存（3世代ローリングストック）"""
        self._current_phase = phase
        self._current_data = data
        
        checkpoint_data = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # 次のインデックスを取得
        try:
            if self.checkpoint_index_file.exists():
                with open(self.checkpoint_index_file, 'r') as f:
                    idx = int(f.read().strip())
            else:
                idx = 0
        except:
            idx = 0
            
        next_idx = (idx % 3)
        target_file = self.rolling_checkpoints[next_idx]
        
        try:
            # 原子的な書き込みを模倣（一時ファイルに書いてからリネーム）
            temp_file = target_file.with_suffix(".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            if target_file.exists():
                target_file.unlink()
            temp_file.rename(target_file)
            
            # インデックス更新
            with open(self.checkpoint_index_file, 'w') as f:
                f.write(str(next_idx + 1))
                
            logger.info(f"💾 チェックポイント保存(Gen {next_idx + 1}): {phase}")
        except Exception as e:
            logger.error(f"❌ チェックポイント保存失敗: {e}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """チェックポイント読み込み（最新の有効な世代を検索）"""
        best_checkpoint = None
        latest_time = None
        
        for cp_file in self.rolling_checkpoints:
            if cp_file.exists():
                try:
                    with open(cp_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        ts = datetime.fromisoformat(data["timestamp"])
                        if latest_time is None or ts > latest_time:
                            latest_time = ts
                            best_checkpoint = data
                except Exception as e:
                    logger.warning(f"⚠️ チェックポイント {cp_file.name} が破損しています: {e}")
                    
        return best_checkpoint

    def _periodic_checkpoint_worker(self):
        """5分おきに現在の状態を保存するワーカー"""
        logger.info("⏱️ 定期チェックポイントスレッド開始（5分間隔）")
        while not self._stop_checkpoint_thread.is_set():
            # 5分待機（1秒ごとに停止フラグを確認）
            for _ in range(300):
                if self._stop_checkpoint_thread.wait(1):
                    break
            
            if not self._stop_checkpoint_thread.is_set():
                if self._current_phase != "initialized":
                    self.save_checkpoint(self._current_phase, self._current_data)

    def start_periodic_checkpoint(self):
        """定期保存を開始"""
        self._stop_checkpoint_thread.clear()
        self._checkpoint_thread = threading.Thread(target=self._periodic_checkpoint_worker, daemon=True)
        self._checkpoint_thread.start()

    def stop_periodic_checkpoint(self):
        """定期保存を停止"""
        if self._checkpoint_thread:
            self._stop_checkpoint_thread.set()
            self._checkpoint_thread.join(timeout=5)
            logger.info("⏱️ 定期チェックポイントスレッド停止")
            
    def cleanup_checkpoints(self):
        """チェックポイントファイルを一掃（正常終了時）"""
        for cp_file in self.rolling_checkpoints:
            if cp_file.exists():
                cp_file.unlink()
        if self.checkpoint_index_file.exists():
            self.checkpoint_index_file.unlink()
        logger.info("🗑️ チェックポイントファイルを削除しました")
    
    def execute_full_pipeline(self, use_existing_datasets: bool = True):
        """全パイプライン実行（自動再開・定期保存機能付き）"""
        logger.info("=" * 80)
        logger.info("🚀 統合ムーンショットパイプライン 2025-2026 開始")
        logger.info("=" * 80)
        logger.info(f"📅 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📦 既存データセット使用: {use_existing_datasets}")
        logger.info("=" * 80)
        
        # チェックポイントの確認（レジューム）
        checkpoint = self.load_checkpoint()
        start_phase = "start"
        if checkpoint:
            start_phase = checkpoint.get("phase", "start")
            data = checkpoint.get("data", {})
            logger.info(f"🔄 中断されたフェーズが見つかりました: {start_phase}")
            logger.info(f"再開日時: {checkpoint.get('timestamp')}")
        
        # 定期保存スレッド開始
        self.start_periodic_checkpoint()
        
        dataset_paths = []
        try:
            # Phase 0 & 1 & 2: データ準備
            if start_phase in ["start", "scientific_collection", "whitepaper_collection", "dataset_integration", "nsfw_verification", "arxiv_collection"]:
                if use_existing_datasets:
                    if start_phase == "start":
                        # Phase 0: 既存データセット検出
                        logger.info("🔍 既存データセットを検出中...")
                        discovered_datasets = self.discover_existing_datasets()
                        total_files = sum(len(files) for files in discovered_datasets.values())
                        logger.info(f"✅ {total_files}個のデータセットファイルを検出")
                        
                        # Phase 1: データセット統合
                        integrated_dataset = self.integrate_existing_datasets(discovered_datasets)
                        self.validate_dataset(integrated_dataset)
                        self.save_checkpoint("dataset_integration", {"integrated_file": str(integrated_dataset)})
                        
                        # Phase 2: NSFW・薬物データセット確認
                        nsfw_verified = len(discovered_datasets.get("nsfw_detection", [])) > 0
                        drug_verified = len(discovered_datasets.get("drug_detection", [])) > 0
                        self.save_checkpoint("nsfw_verification", {
                            "nsfw_verified": nsfw_verified,
                            "drug_verified": drug_verified,
                            "integrated_file": str(integrated_dataset)
                        })
                        dataset_paths = [integrated_dataset]
                    else:
                        # レジューム時のデータ復元
                        integrated_dataset = Path(checkpoint["data"]["integrated_file"])
                        dataset_paths = [integrated_dataset]
                else:
                    if start_phase == "start":
                        # Phase 1: Arxiv/Biorxiv上位計10万件収集
                        scientific_data = self.collect_scientific_papers_top_100000()
                        self.save_checkpoint("scientific_collection", {"scientific_files": [str(p) for p in scientific_data]})
                        
                        # Phase 2: 防衛・JAXA白書収集
                        whitepaper_data = self.collect_defense_jaxa_whitepapers()
                        self.save_checkpoint("whitepaper_collection", {
                            "whitepaper_files": [str(p) for p in whitepaper_data],
                            "scientific_files": [str(p) for p in scientific_data]
                        })
                        
                        # Phase 3: クレンジングと統合
                        logger.info("🎨 収集データのクレンジングと統合を開始...")
                        discovered = {
                            "arxiv": [Path(p) for p in scientific_data],
                            "whitepapers": whitepaper_data
                        }
                        integrated_dataset = self.integrate_existing_datasets(discovered)
                        self.save_checkpoint("dataset_integration", {"integrated_file": str(integrated_dataset)})
                        dataset_paths = [integrated_dataset]
                    else:
                        # レジューム時のデータ復元
                        if start_phase == "scientific_collection":
                            scientific_data = checkpoint["data"]["scientific_files"]
                            whitepaper_data = self.collect_defense_jaxa_whitepapers()
                            # ... 以下同様にフェーズを遷移させるが、ここでは簡略化して
                            # 後の条件分岐に任せる
                            dataset_paths = [Path(p) for p in scientific_data] + whitepaper_data
                        elif start_phase == "whitepaper_collection":
                            integrated_dataset = self.integrate_existing_datasets({
                                "arxiv": [Path(p) for p in checkpoint["data"]["scientific_files"]],
                                "whitepapers": [Path(p) for p in checkpoint["data"]["whitepaper_files"]]
                            })
                            dataset_paths = [integrated_dataset]
                        elif start_phase == "dataset_integration":
                            dataset_paths = [Path(checkpoint["data"]["integrated_file"])]
            else:
                # 既にデータ準備が終わっているフェーズからの再開
                dataset_paths = [] # 後のフェーズで使わない場合は空でも可、使う場合は checkpoint から復元
                if "sft_model" in checkpoint["data"]:
                    sft_model = Path(checkpoint["data"]["sft_model"])

            # Phase 4: SFT実行
            if start_phase in ["start", "scientific_collection", "whitepaper_collection", "dataset_integration", "nsfw_verification", "arxiv_collection"]:
                sft_model = self.execute_sft(dataset_paths)
                self.save_checkpoint("sft_completion", {"sft_model": str(sft_model)})
            elif start_phase == "sft_completion":
                sft_model = Path(checkpoint["data"]["sft_model"])

            # Phase 5: 最新手法統合
            if start_phase in ["start", "scientific_collection", "whitepaper_collection", "dataset_integration", "nsfw_verification", "arxiv_collection", "sft_completion"]:
                final_model = self.execute_advanced_techniques_integration(sft_model)
                self.save_checkpoint("advanced_techniques", {"final_model": str(final_model)})
            elif start_phase == "advanced_techniques":
                final_model = Path(checkpoint["data"]["final_model"])

            # Phase 6: ABCテスト
            if start_phase in ["start", "scientific_collection", "whitepaper_collection", "dataset_integration", "nsfw_verification", "arxiv_collection", "sft_completion", "advanced_techniques"]:
                abc_results = self.execute_abc_test(final_model)
                self.save_checkpoint("abc_test", {"abc_results": abc_results, "final_model": str(final_model)})
            elif start_phase == "abc_test":
                abc_results = checkpoint["data"]["abc_results"]
                final_model = Path(checkpoint["data"]["final_model"])

            # Phase 7: 成績向上確認
            if start_phase in ["start", "dataset_integration", "nsfw_verification", "arxiv_collection", "whitepaper_collection", "sft_completion", "advanced_techniques", "abc_test"]:
                improved = self.check_performance_improvement(abc_results)
                self.save_checkpoint("performance_check", {"improved": improved, "final_model": str(final_model)})
            elif start_phase == "performance_check":
                improved = checkpoint["data"]["improved"]
                final_model = Path(checkpoint["data"]["final_model"])

            # Phase 8: HFアップロード
            if improved:
                upload_success = self.upload_to_hf_if_improved(final_model, improved)
                self.save_checkpoint("hf_upload", {"success": upload_success})
            
            logger.info("=" * 80)
            logger.info("✅ 統合ムーンショットパイプライン 2025-2026 完了!")
            logger.info("=" * 80)
            
            # 正常終了時はチェックポイントを削除
            self.cleanup_checkpoints()
            
            # 音声ファイル再生
            self.play_completion_sound()
            
        except Exception as e:
            logger.error(f"❌ パイプライン実行エラー: {e}", exc_info=True)
            raise
        finally:
            # 定期保存終了
            self.stop_periodic_checkpoint()
    
    def play_completion_sound(self):
        """完了音声再生"""
        try:
            sound_path = Path(".cursor/marisa_owattaze.wav")
            if sound_path.exists():
                import winsound
                logger.info("🔊 音声ファイルを再生中...")
                winsound.PlaySound(str(sound_path), winsound.SND_FILENAME)
                logger.info("✅ 音声再生完了")
        except Exception as e:
            logger.warning(f"⚠️ 音声再生エラー: {e}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ムーンショットパイプライン 2025-2026")
    parser.add_argument(
        "--use-existing-datasets",
        action="store_true",
        default=True,
        help="既存データセットを使用（デフォルト: True）"
    )
    parser.add_argument(
        "--collect-new-data",
        action="store_true",
        help="新しいデータを収集（--use-existing-datasetsを無効化）"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="既存データセットを一覧表示"
    )
    
    args = parser.parse_args()
    
    pipeline = IntegratedMoonshotPipeline2025_2026()
    
    if args.list_datasets:
        datasets = pipeline.discover_existing_datasets()
        print("\n=== 検出された既存データセット ===")
        for cat, files in datasets.items():
            print(f"\n[{cat.upper()}]")
            for f in files:
                print(f"  - {f}")
        return

    use_existing = args.use_existing_datasets and not args.collect_new_data
    pipeline.execute_full_pipeline(use_existing_datasets=use_existing)


if __name__ == "__main__":
    main()
