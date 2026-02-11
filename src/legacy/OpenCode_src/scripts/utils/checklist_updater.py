#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
チェックリスト自動更新ユーティリティ

_docs/progress_checklist.mdを自動更新し、フェーズ完了時にチェックマークを付与

Usage:
    from scripts.utils.checklist_updater import ChecklistUpdater
    
    updater = ChecklistUpdater()
    updater.update_phase_completion("phase1", metrics={"accuracy": 0.95})
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChecklistUpdater:
    """チェックリスト更新メインクラス"""
    
    def __init__(self, checklist_path: Optional[Path] = None):
        """
        Args:
            checklist_path: チェックリストファイルパス（デフォルト: _docs/progress_checklist.md）
        """
        self.checklist_path = checklist_path or (PROJECT_ROOT / "_docs" / "progress_checklist.md")
        self.checklist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # チェックリストが存在しない場合は作成
        if not self.checklist_path.exists():
            self._create_initial_checklist()
        
        logger.info("="*80)
        logger.info("Checklist Updater Initialized")
        logger.info("="*80)
        logger.info(f"Checklist path: {self.checklist_path}")
    
    def _create_initial_checklist(self):
        """初期チェックリストを作成"""
        content = """# SO8T Complete Pipeline Progress Checklist

## 実装状況

### Phase 1: データ収集・前処理パイプライン
- [ ] Webスクレイピング
- [ ] 統計的データクレンジング
- [ ] クラス分類自動化
- **状態**: 未開始
- **開始日時**: -
- **完了日時**: -
- **実行時間**: -
- **メトリクス**: -

### Phase 2: SO(8) Transformer再学習
- [ ] SO(8) Transformerモデル構築
- [ ] ベイズ最適化統合
- [ ] 電源断リカバリー
- **状態**: 未開始
- **開始日時**: -
- **完了日時**: -
- **実行時間**: -
- **メトリクス**: -

### Phase 3: GGUF変換（A/Bモデル）
- [ ] モデルA（最適化なし）GGUF変換
- [ ] モデルB（ベイズ最適化済み）GGUF変換
- **状態**: 未開始
- **開始日時**: -
- **完了日時**: -
- **実行時間**: -
- **メトリクス**: -

### Phase 4: A/Bテスト評価
- [ ] A/Bテスト実行
- [ ] HFベンチマークテスト
- **状態**: 未開始
- **開始日時**: -
- **完了日時**: -
- **実行時間**: -
- **メトリクス**: -

### Phase 5: 可視化・レポート生成
- [ ] A/Bテスト結果可視化
- **状態**: 未開始
- **開始日時**: -
- **完了日時**: -
- **実行時間**: -
- **メトリクス**: -

### Phase 6: 統合パイプライン
- [ ] 全フェーズ統合実行
- [ ] チェックポイント管理
- [ ] エラーハンドリング
- **状態**: 未開始
- **開始日時**: -
- **完了日時**: -
- **実行時間**: -
- **メトリクス**: -

### Phase 7: 全自動化スクリプト
- [ ] Windows起動時自動実行設定
- [ ] 前回セッションからの自動復旧
- [ ] 進捗管理システム統合
- **状態**: 未開始
- **開始日時**: -
- **完了日時**: -
- **実行時間**: -
- **メトリクス**: -

## 進捗サマリー

- **総フェーズ数**: 7
- **完了フェーズ数**: 0
- **実行中フェーズ数**: 0
- **失敗フェーズ数**: 0
- **全体進捗**: 0.0%

---
*最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(self.checklist_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Initial checklist created: {self.checklist_path}")
    
    def update_phase_completion(
        self,
        phase_name: str,
        status: str = "completed",
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """
        フェーズ完了を更新
        
        Args:
            phase_name: フェーズ名（phase1, phase2, ...）
            status: 状態（completed, failed, running）
            metrics: メトリクス辞書
            error_message: エラーメッセージ（失敗時）
        """
        # チェックリストを読み込み
        content = self.checklist_path.read_text(encoding='utf-8')
        
        # phase_nameから番号を抽出（例: "phase1" -> "1"）
        phase_num_match = re.search(r'phase(\d+)', phase_name.lower())
        if not phase_num_match:
            logger.warning(f"Invalid phase name format: {phase_name}")
            return
        
        phase_number = phase_num_match.group(1)
        
        # 該当するフェーズセクションを検索
        phase_pattern = rf"### Phase {phase_number}: .+?\n(.*?)(?=### Phase|\n## |$)"
        phase_match = re.search(phase_pattern, content, re.DOTALL)
        
        if not phase_match:
            logger.warning(f"Phase {phase_number} section not found in checklist")
            return
        
        phase_section = phase_match.group(1)
        
        # チェックマークを更新
        if status == "completed":
            phase_section = re.sub(r'- \[ \]', '- [x]', phase_section)
        elif status == "failed":
            phase_section = re.sub(r'- \[ \]', '- [x]', phase_section)
            phase_section = re.sub(r'- \[x\]', '- [x]', phase_section)
        
        # 状態を更新
        status_emoji = {
            'completed': '✅ 完了',
            'failed': '❌ 失敗',
            'running': '🔄 実行中',
            'pending': '⏳ 未開始'
        }.get(status, status)
        
        phase_section = re.sub(
            r'\*\*状態\*\*: .+',
            f'**状態**: {status_emoji}',
            phase_section
        )
        
        # 日時を更新
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if status == "running":
            # 開始日時を更新
            phase_section = re.sub(
                r'\*\*開始日時\*\*: .+',
                f'**開始日時**: {now}',
                phase_section
            )
        elif status in ["completed", "failed"]:
            # 完了日時を更新
            phase_section = re.sub(
                r'\*\*完了日時\*\*: .+',
                f'**完了日時**: {now}',
                phase_section
            )
            
            # 実行時間を計算（開始日時から）
            start_match = re.search(r'\*\*開始日時\*\*: (.+)', phase_section)
            if start_match:
                try:
                    start_time = datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.now()
                    duration = end_time - start_time
                    duration_str = self._format_duration(duration)
                    phase_section = re.sub(
                        r'\*\*実行時間\*\*: .+',
                        f'**実行時間**: {duration_str}',
                        phase_section
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate duration: {e}")
        
        # メトリクスを更新
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                    for k, v in metrics.items()])
            phase_section = re.sub(
                r'\*\*メトリクス\*\*: .+',
                f'**メトリクス**: {metrics_str}',
                phase_section
            )
        
        # エラーメッセージを追加
        if error_message:
            if "**エラー**: " not in phase_section:
                phase_section += f"\n- **エラー**: {error_message}\n"
            else:
                phase_section = re.sub(
                    r'\*\*エラー\*\*: .+',
                    f'**エラー**: {error_message}',
                    phase_section
                )
        
        # セクションを置換
        content = content[:phase_match.start(1)] + phase_section + content[phase_match.end(1):]
        
        # サマリーを更新
        content = self._update_summary(content)
        
        # 最終更新日時を更新
        content = re.sub(
            r'\*最終更新: .+\*',
            f'*最終更新: {now}*',
            content
        )
        
        # 保存
        self.checklist_path.write_text(content, encoding='utf-8')
        logger.info(f"Checklist updated for {phase_name} (status: {status})")
    
    def _format_duration(self, duration) -> str:
        """実行時間をフォーマット"""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}時間{minutes}分{seconds}秒"
        elif minutes > 0:
            return f"{minutes}分{seconds}秒"
        else:
            return f"{seconds}秒"
    
    def _update_summary(self, content: str) -> str:
        """進捗サマリーを更新"""
        # 各フェーズの状態を取得
        phases = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6', 'phase7']
        completed = 0
        running = 0
        failed = 0
        
        for phase in phases:
            phase_pattern = rf"### Phase \d+: .+?\n.*?\*\*状態\*\*: (.+?)\n"
            match = re.search(phase_pattern, content, re.DOTALL)
            if match:
                status = match.group(1)
                if '完了' in status:
                    completed += 1
                elif '実行中' in status:
                    running += 1
                elif '失敗' in status:
                    failed += 1
        
        total = len(phases)
        overall_progress = (completed / total * 100) if total > 0 else 0.0
        
        # サマリーセクションを更新
        summary_pattern = r'## 進捗サマリー\n\n(.*?)\n\n---'
        summary_content = f"""## 進捗サマリー

- **総フェーズ数**: {total}
- **完了フェーズ数**: {completed}
- **実行中フェーズ数**: {running}
- **失敗フェーズ数**: {failed}
- **全体進捗**: {overall_progress:.1f}%"""
        
        content = re.sub(summary_pattern, summary_content + '\n\n---', content, flags=re.DOTALL)
        
        return content
    
    def add_phase_metrics(self, phase_name: str, metrics: Dict[str, Any]):
        """フェーズメトリクスを追加"""
        self.update_phase_completion(phase_name, status="completed", metrics=metrics)
    
    def generate_checklist(self) -> str:
        """チェックリストを生成（現在の内容を返す）"""
        return self.checklist_path.read_text(encoding='utf-8')


def main():
    """テスト用メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Checklist Updater Test")
    parser.add_argument("--phase", type=str, default="phase1", help="Phase name")
    parser.add_argument("--status", type=str, default="completed", help="Status")
    args = parser.parse_args()
    
    updater = ChecklistUpdater()
    updater.update_phase_completion(
        args.phase,
        status=args.status,
        metrics={"accuracy": 0.95, "f1_score": 0.92}
    )
    
    print("\nChecklist updated successfully!")
    print(f"Checklist path: {updater.checklist_path}")


if __name__ == "__main__":
    main()

