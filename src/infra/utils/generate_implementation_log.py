"""
実装ログ自動生成スクリプト

Git worktree名と日付を自動取得し、実装ログテンプレートを生成する。
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys


def get_worktree_name() -> str:
    """
    現在のgit worktree名を取得
    
    Returns:
        worktree名（worktreeでない場合は"main"）
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True
        )
        git_dir = result.stdout.strip()
        
        if "worktrees" in git_dir:
            # Extract worktree name from path
            # Format: C:/path/to/.git/worktrees/{worktree_name}
            parts = Path(git_dir).parts
            if "worktrees" in parts:
                idx = parts.index("worktrees")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        return "main"
    except Exception as e:
        print(f"[WARNING] Failed to get worktree name: {e}")
        return "main"


def generate_log_template(
    feature_name: str,
    worktree_name: str,
    date: str,
    author: str = "AI Agent",
) -> str:
    """
    実装ログテンプレートを生成
    
    Args:
        feature_name: 機能名
        worktree_name: worktree名
        date: 日付（YYYY-MM-DD形式）
        author: 実装者名
    
    Returns:
        実装ログテンプレート（Markdown形式）
    """
    template = f"""# {feature_name} 実装ログ

## 実装情報
- **日付**: {date}
- **Worktree**: {worktree_name}
- **機能名**: {feature_name}
- **実装者**: {author}

## 概要

{feature_name}の実装内容を記録します。

## 実装内容

### 1. [実装項目1]

**ファイル**: `path/to/file.py`

**実装状況**: [実装済み] / [未実装] / [部分実装]  
**動作確認**: [OK] / [要修正] / [未確認]  
**確認日時**: YYYY-MM-DD（該当する場合）  
**備考**: 簡潔なメモ（該当する場合）

- 実装内容の説明

## 作成・変更ファイル

### 新規作成ファイル

1. **カテゴリ1**:
   - `path/to/file1.py`
   - `path/to/file2.py`

### 変更ファイル

1. **カテゴリ1**:
   - `path/to/file1.py`: 変更内容の説明

## 設計判断

### 1. [設計判断1]

**理由**:
- 理由1
- 理由2

**利点**:
- 利点1
- 利点2

## テスト結果

### 実装完了項目

- [ ] 項目1
- [ ] 項目2

### リンターエラー

- エラー状況

## 今後の拡張予定

1. **拡張項目1**:
   - 説明

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

## 参考資料

- 実装計画: `cursor-plan://...`
- 関連ドキュメント: `_docs/...`

---

**実装完了日時**: {date}  
**Worktree**: {worktree_name}  
**実装者**: {author}
"""
    return template


def main():
    parser = argparse.ArgumentParser(
        description="Generate implementation log template"
    )
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help="Feature name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_docs"),
        help="Output directory (default: _docs)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date (YYYY-MM-DD format, default: today)",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="AI Agent",
        help="Author name (default: AI Agent)",
    )
    parser.add_argument(
        "--worktree",
        type=str,
        help="Worktree name (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # 日付を取得
    if args.date:
        date = args.date
    else:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Worktree名を取得
    if args.worktree:
        worktree_name = args.worktree
    else:
        worktree_name = get_worktree_name()
    
    # ファイル名を生成
    feature_slug = args.feature.lower().replace(" ", "_")
    filename = f"{date}_{worktree_name}_{feature_slug}.md"
    output_path = args.output / filename
    
    # テンプレートを生成
    template = generate_log_template(
        feature_name=args.feature,
        worktree_name=worktree_name,
        date=date,
        author=args.author,
    )
    
    # ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ファイルに書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    print(f"[SUCCESS] Implementation log created: {output_path}")
    print(f"[INFO] Worktree: {worktree_name}")
    print(f"[INFO] Date: {date}")
    print(f"[INFO] Feature: {args.feature}")


if __name__ == "__main__":
    main()

