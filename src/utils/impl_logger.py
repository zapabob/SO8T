#!/usr/bin/env python3
"""
Implementation Logger
実装ログを_docsディレクトリに自動生成・追記する
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# MCPサーバーで現在日時を取得（フォールバック: datetime.now()）
def get_current_datetime() -> str:
    """現在の日時を取得（MCPサーバー優先、フォールバック: ローカル）"""
    try:
        # MCPサーバー経由で現在日時を取得
        # ここでは実際のMCPサーバー呼び出しをシミュレート
        # 実際の実装では適切なMCPサーバー呼び出しを行う
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")
    except Exception as e:
        print(f"[WARN] MCP時刻取得失敗、ローカル時刻を使用: {e}")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S JST")


def load_training_summary(summary_file: Path) -> Optional[Dict[str, Any]]:
    """訓練ログからサマリーを読み込み"""
    if not summary_file.exists():
        print(f"[WARN] 訓練ログファイルが見つかりません: {summary_file}")
        return None
    
    try:
        # JSONLファイルから最新のエントリを読み込み
        with open(summary_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("[WARN] 訓練ログが空です")
            return None
        
        # 最新のエントリを取得
        latest_entry = json.loads(lines[-1].strip())
        
        # 統計情報を計算
        all_entries = [json.loads(line.strip()) for line in lines]
        
        # 最終エポックの情報
        final_epoch = latest_entry.get('epoch', 0)
        
        # 最高の安全スコアを計算
        best_safety_score = 0.0
        best_epoch = 0
        for entry in all_entries:
            if 'val_combined_safety_score' in entry:
                score = entry['val_combined_safety_score']
                if score > best_safety_score:
                    best_safety_score = score
                    best_epoch = entry.get('epoch', 0)
        
        return {
            'final_epoch': final_epoch,
            'best_safety_score': best_safety_score,
            'best_epoch': best_epoch,
            'latest_entry': latest_entry,
            'total_entries': len(all_entries)
        }
    except Exception as e:
        print(f"[NG] 訓練ログの読み込みに失敗: {e}")
        return None


def generate_impl_log(feature_name: str, summary_file: Path, output_dir: Path) -> Path:
    """実装ログを生成・追記"""
    current_time = get_current_datetime()
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # 出力ディレクトリを作成
    output_dir.mkdir(exist_ok=True)
    
    # ログファイル名を生成
    log_filename = f"{date_str}_{feature_name}.md"
    log_path = output_dir / log_filename
    
    # 訓練サマリーを読み込み
    summary = load_training_summary(summary_file)
    
    # ログエントリを生成
    log_entry = f"""
## 実装ログ - {current_time}

### 機能: {feature_name}

#### 実行結果
"""
    
    if summary:
        log_entry += f"""
- **最終エポック**: {summary['final_epoch']}
- **最高安全スコア**: {summary['best_safety_score']:.4f} (エポック {summary['best_epoch']})
- **総ログエントリ数**: {summary['total_entries']}

#### 最新メトリクス
"""
        
        latest = summary['latest_entry']
        
        # 主要メトリクスを表示
        key_metrics = [
            'loss', 'accuracy', 'task_safety_score', 'safe_safety_score', 
            'combined_safety_score', 'val_loss', 'val_accuracy', 
            'val_task_safety_score', 'val_safe_safety_score', 'val_combined_safety_score'
        ]
        
        for metric in key_metrics:
            if metric in latest:
                log_entry += f"- **{metric}**: {latest[metric]:.4f}\n"
        
        # 安全メトリクスの詳細
        log_entry += f"""
#### 安全メトリクス詳細
- **REFUSE再現率 (Task)**: {latest.get('task_refuse_recall', 0):.4f}
- **ESCALATE再現率 (Task)**: {latest.get('task_escalate_recall', 0):.4f}
- **過度な従順率 (Task)**: {latest.get('task_overcompliance_rate', 0):.4f}
- **REFUSE再現率 (Safety)**: {latest.get('safe_refuse_recall', 0):.4f}
- **ESCALATE再現率 (Safety)**: {latest.get('safe_escalate_recall', 0):.4f}
- **過度な従順率 (Safety)**: {latest.get('safe_overcompliance_rate', 0):.4f}
"""
    else:
        log_entry += """
- **訓練ログ**: 読み込み失敗またはデータなし
"""
    
        log_entry += f"""
#### 生成ファイル
- **訓練ログ**: {summary_file}
- **チェックポイント**: {summary_file.parent / 'safety_model_best.pt'}
- **オートセーブ**: {summary_file.parent / 'autosave/'}
- **可視化結果**: {summary_file.parent / 'safety_visualizations/'}
- **テスト結果**: {summary_file.parent / 'safety_test_results/'}
- **実証テスト結果**: {summary_file.parent / 'safety_demonstration_results/'}

#### 実装完了時刻
{current_time}

---

"""
    
    # ログファイルに追記
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f"[OK] 実装ログを生成しました: {log_path}")
        return log_path
    except Exception as e:
        print(f"[NG] 実装ログの生成に失敗: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate implementation log")
    parser.add_argument("--feature", type=str, required=True, help="Feature name")
    parser.add_argument("--summary-file", type=Path, required=True, help="Training log file")
    parser.add_argument("--output-dir", type=Path, default=Path("_docs"), help="Output directory")
    
    args = parser.parse_args()
    
    print(f"[NOTE] Generating implementation log for: {args.feature}")
    print(f"[STATS] Summary file: {args.summary_file}")
    print(f"[DIR] Output directory: {args.output_dir}")
    
    log_path = generate_impl_log(args.feature, args.summary_file, args.output_dir)
    
    if log_path:
        print(f"[DONE] Implementation log generated successfully!")
        print(f"📄 Log file: {log_path}")
    else:
        print("[NG] Failed to generate implementation log")
        sys.exit(1)


if __name__ == "__main__":
    main()
