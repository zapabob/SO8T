"""
実装ログ更新ヘルパースクリプト

既存の実装ログの進捗状況（実装状況・動作確認結果）を更新する。
"""

import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


def find_item_section(content: str, item_number: int) -> Tuple[Optional[int], Optional[int]]:
    """
    指定された項目番号のセクションを検索
    
    Args:
        content: ファイル内容
        item_number: 項目番号（1-11など）
    
    Returns:
        (開始行番号, 終了行番号) のタプル（見つからない場合はNone）
    """
    pattern = rf"^### {item_number}\.\s+"
    lines = content.split('\n')
    
    start_line = None
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            start_line = i
            break
    
    if start_line is None:
        return None, None
    
    # 次の項目セクションまたは次の主要セクションまでを終了位置とする
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        if re.match(r"^### \d+\.\s+", lines[i]) or \
           re.match(r"^## [^#]", lines[i]):
            end_line = i
            break
    
    return start_line, end_line


def update_item_status(
    content: str,
    item_number: int,
    status: Optional[str] = None,
    verification: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    指定された項目の進捗状況を更新
    
    Args:
        content: ファイル内容
        item_number: 項目番号
        status: 実装状況（"実装済み", "未実装", "部分実装"）
        verification: 動作確認（"OK", "要修正", "未確認"）
        notes: 備考
    
    Returns:
        更新後のファイル内容
    """
    start_line, end_line = find_item_section(content, item_number)
    
    if start_line is None:
        print(f"[WARNING] Item {item_number} not found in log file")
        return content
    
    lines = content.split('\n')
    section_lines = lines[start_line:end_line]
    section_content = '\n'.join(section_lines)
    
    # 既存の実装状況・動作確認・確認日時・備考を検索
    status_pattern = r"\*\*実装状況\*\*:\s*\[([^\]]+)\]"
    verification_pattern = r"\*\*動作確認\*\*:\s*\[([^\]]+)\]"
    date_pattern = r"\*\*確認日時\*\*:\s*([^\n]+)"
    notes_pattern = r"\*\*備考\*\*:\s*([^\n]+)"
    
    # 更新
    today = datetime.now().strftime("%Y-%m-%d")
    
    if status:
        if re.search(status_pattern, section_content):
            section_content = re.sub(
                status_pattern,
                f"**実装状況**: [{status}]",
                section_content
            )
        else:
            # 実装状況行が存在しない場合は追加
            file_line = next((i for i, line in enumerate(section_lines) if "**ファイル**:" in line), None)
            if file_line is not None:
                insert_pos = file_line + 1
                section_lines.insert(insert_pos, f"**実装状況**: [{status}]")
                section_content = '\n'.join(section_lines)
    
    if verification:
        if re.search(verification_pattern, section_content):
            section_content = re.sub(
                verification_pattern,
                f"**動作確認**: [{verification}]",
                section_content
            )
            # 確認日時も更新
            if verification in ["OK", "要修正"]:
                if re.search(date_pattern, section_content):
                    section_content = re.sub(
                        date_pattern,
                        f"**確認日時**: {today}",
                        section_content
                    )
                else:
                    # 確認日時行が存在しない場合は追加
                    section_content = re.sub(
                        r"(\*\*動作確認\*\*:\s*\[[^\]]+\])",
                        f"\\1\n**確認日時**: {today}",
                        section_content
                    )
        else:
            # 動作確認行が存在しない場合は追加
            status_line = next((i for i, line in enumerate(section_lines) if "**実装状況**:" in line), None)
            if status_line is not None:
                insert_pos = status_line + 1
                section_lines.insert(insert_pos, f"**動作確認**: [{verification}]")
                if verification in ["OK", "要修正"]:
                    section_lines.insert(insert_pos + 1, f"**確認日時**: {today}")
                section_content = '\n'.join(section_lines)
    
    if notes:
        if re.search(notes_pattern, section_content):
            section_content = re.sub(
                notes_pattern,
                f"**備考**: {notes}",
                section_content
            )
        else:
            # 備考行が存在しない場合は追加
            date_line = next((i for i, line in enumerate(section_lines) if "**確認日時**:" in line), None)
            if date_line is not None:
                insert_pos = date_line + 1
            else:
                verification_line = next((i for i, line in enumerate(section_lines) if "**動作確認**:" in line), None)
                if verification_line is not None:
                    insert_pos = verification_line + 1
                else:
                    status_line = next((i for i, line in enumerate(section_lines) if "**実装状況**:" in line), None)
                    if status_line is not None:
                        insert_pos = status_line + 1
                    else:
                        insert_pos = 1
            
            section_lines.insert(insert_pos, f"**備考**: {notes}")
            section_content = '\n'.join(section_lines)
    
    # 元の内容に反映
    updated_lines = lines[:start_line] + section_content.split('\n') + lines[end_line:]
    return '\n'.join(updated_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Update implementation log progress"
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Implementation log file path",
    )
    parser.add_argument(
        "--item",
        type=int,
        required=True,
        help="Item number (1-11, etc.)",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["実装済み", "未実装", "部分実装"],
        help="Implementation status",
    )
    parser.add_argument(
        "--verification",
        type=str,
        choices=["OK", "要修正", "未確認"],
        help="Verification status",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes",
    )
    
    args = parser.parse_args()
    
    if not args.log.exists():
        print(f"[ERROR] Log file not found: {args.log}")
        return
    
    # ファイルを読み込み
    with open(args.log, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 更新
    updated_content = update_item_status(
        content,
        item_number=args.item,
        status=args.status,
        verification=args.verification,
        notes=args.notes,
    )
    
    # ファイルに書き込み
    with open(args.log, "w", encoding="utf-8") as f:
        f.write(updated_content)
    
    print(f"[SUCCESS] Updated item {args.item} in {args.log}")
    if args.status:
        print(f"[INFO] Status: {args.status}")
    if args.verification:
        print(f"[INFO] Verification: {args.verification}")
    if args.notes:
        print(f"[INFO] Notes: {args.notes}")


if __name__ == "__main__":
    main()

