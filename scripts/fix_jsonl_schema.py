#!/usr/bin/env python3
"""
Fix JSONL Schema - Convert JSON arrays to proper JSONL format
Handles files with JSON arrays instead of line-delimited JSON objects
"""

import json
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_jsonl_file(
    input_path: Path, output_path: Path = None, backup: bool = True
) -> bool:
    """
    Convert JSON array file to proper JSONL format

    Args:
        input_path: Path to input file (JSON array format)
        output_path: Path to output file (JSONL format). If None, overwrites input
        backup: Whether to create backup of original file

    Returns:
        bool: True if successful, False otherwise
    """
    if not input_path.exists():
        logger.error(f"[ERROR] File not found: {input_path}")
        return False

    if output_path is None:
        output_path = input_path

    # Create backup if requested and overwriting
    if backup and output_path == input_path:
        backup_path = input_path.with_suffix(".jsonl.backup")
        try:
            input_path.rename(backup_path)
            input_path = backup_path
            logger.info(f"[BACKUP] Created backup: {backup_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to create backup: {e}")
            return False

    try:
        # Read with BOM handling for UTF-8-SIG
        with open(input_path, "r", encoding="utf-8-sig") as f:
            content = f.read().strip()

            if not content:
                logger.warning(f"[WARN] Empty file: {input_path}")
                return False

            # Try parsing as JSON array
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    logger.error(
                        f"[ERROR] File {input_path} is not a JSON array (type: {type(data).__name__})"
                    )
                    return False
            except json.JSONDecodeError as e:
                logger.error(f"[ERROR] Failed to parse {input_path} as JSON: {e}")
                # Try line-by-line parsing as fallback
                return try_fix_line_by_line(input_path, output_path)

        # Write as JSONL
        fixed_count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                fixed_count += 1

        logger.info(
            f"[OK] Fixed {input_path.name}: {fixed_count} records -> {output_path}"
        )

        # Remove backup if successful and different output
        if backup and input_path != output_path and input_path.suffix == ".backup":
            input_path.unlink()
            logger.info(f"[CLEANUP] Removed backup: {input_path}")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Unexpected error fixing {input_path}: {e}")
        return False


def try_fix_line_by_line(input_path: Path, output_path: Path) -> bool:
    """
    Fallback: Try to parse file line by line, fixing each line if possible
    """
    logger.info(f"[FALLBACK] Trying line-by-line fix for {input_path}")

    fixed_records = []
    error_count = 0

    try:
        with open(input_path, "r", encoding="utf-8-sig") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Try parsing as JSON
                    record = json.loads(line)
                    fixed_records.append(record)
                except json.JSONDecodeError:
                    # Try extracting JSON from the line
                    try:
                        # Handle case where line might be a JSON array element
                        if line.startswith("{") and line.endswith("}"):
                            record = json.loads(line)
                            fixed_records.append(record)
                        else:
                            error_count += 1
                            if error_count <= 5:
                                logger.warning(
                                    f"[WARN] Line {line_num}: Could not parse - {line[:100]}..."
                                )
                    except:
                        error_count += 1

        if fixed_records:
            with open(output_path, "w", encoding="utf-8") as f:
                for record in fixed_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(
                f"[OK] Line-by-line fix successful: {len(fixed_records)} records, {error_count} errors"
            )
            return True
        else:
            logger.error(f"[ERROR] No valid records found in {input_path}")
            return False

    except Exception as e:
        logger.error(f"[ERROR] Line-by-line fix failed: {e}")
        return False


def scan_and_fix_directory(directory: Path, pattern: str = "*.jsonl") -> dict:
    """
    Scan directory for JSONL files and fix those with array format

    Returns:
        dict: Summary of fixes {'fixed': [], 'failed': [], 'skipped': []}
    """
    results = {"fixed": [], "failed": [], "skipped": []}

    if not directory.exists():
        logger.error(f"[ERROR] Directory not found: {directory}")
        return results

    jsonl_files = list(directory.glob(pattern))
    logger.info(f"[SCAN] Found {len(jsonl_files)} JSONL files in {directory}")

    for file_path in jsonl_files:
        # Check if file needs fixing (contains array format)
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                first_line = f.readline().strip()

                # If starts with '[', it's likely a JSON array
                if first_line.startswith("["):
                    logger.info(f"[DETECT] Array format detected: {file_path.name}")
                    if fix_jsonl_file(file_path):
                        results["fixed"].append(str(file_path))
                    else:
                        results["failed"].append(str(file_path))
                else:
                    results["skipped"].append(str(file_path))
        except Exception as e:
            logger.error(f"[ERROR] Could not check {file_path}: {e}")
            results["failed"].append(str(file_path))

    return results


def main():
    """Main entry point with command line argument support"""
    import argparse

    parser = argparse.ArgumentParser(description="Fix JSONL schema issues")
    parser.add_argument("--file", "-f", type=str, help="Single file to fix")
    parser.add_argument("--directory", "-d", type=str, help="Directory to scan")
    parser.add_argument(
        "--pattern", "-p", type=str, default="*.jsonl", help="File pattern"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file (for single file mode)"
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")

    args = parser.parse_args()

    if args.file:
        # Single file mode
        input_path = Path(args.file)
        output_path = Path(args.output) if args.output else None
        success = fix_jsonl_file(input_path, output_path, backup=not args.no_backup)
        sys.exit(0 if success else 1)

    elif args.directory:
        # Directory scan mode
        directory = Path(args.directory)
        results = scan_and_fix_directory(directory, args.pattern)

        print("\n" + "=" * 60)
        print("FIX SUMMARY")
        print("=" * 60)
        print(f"Fixed:   {len(results['fixed'])} files")
        print(f"Failed:  {len(results['failed'])} files")
        print(f"Skipped: {len(results['skipped'])} files (already valid)")

        if results["fixed"]:
            print("\n[FIXED FILES]")
            for f in results["fixed"]:
                print(f"  - {f}")

        if results["failed"]:
            print("\n[FAILED FILES]")
            for f in results["failed"]:
                print(f"  - {f}")

        sys.exit(0 if not results["failed"] else 1)

    else:
        # Default: Fix known problematic files
        files_to_fix = [
            Path(
                "H:/from_D/webdataset/phi35_integrated/phi35_ppo_optimized_integrated.jsonl"
            ),
            Path(
                "H:/from_D/webdataset/datasets/soul_weights/soul_weights_dataset.jsonl"
            ),
            Path(
                "H:/from_D/webdataset/datasets/shi3z_anthropic_hh_rlhf_japanese/arlhfj.jsonl"
            ),
        ]

        print("=" * 60)
        print("JSONL Schema Fix - Default Mode")
        print("=" * 60)

        success_count = 0
        for file_path in files_to_fix:
            if file_path.exists():
                if fix_jsonl_file(file_path):
                    success_count += 1
            else:
                logger.warning(f"[SKIP] File not found: {file_path}")

        print(f"\n[SUMMARY] Fixed {success_count}/{len(files_to_fix)} files")
        sys.exit(0 if success_count == len(files_to_fix) else 1)


if __name__ == "__main__":
    main()
