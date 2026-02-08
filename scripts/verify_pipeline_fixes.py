#!/usr/bin/env python3
"""
AEGIS Pipeline Verification Script
Verifies all fixes have been applied correctly
"""

import subprocess
import sys
from pathlib import Path
import json


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_status(status, message, detail=""):
    """Print status line"""
    symbol = "[OK]" if status else "[NG]"
    print(f"  {symbol} {message}")
    if detail and not status:
        print(f"      -> {detail}")


def verify_jsonl_fixes():
    """Verify JSONL files are in correct format"""
    print_header("Phase 1: JSONL Schema Fixes")

    jsonl_files = [
        (
            "H:/from_D/webdataset/phi35_integrated/phi35_ppo_optimized_integrated.jsonl",
            80000,
        ),
        ("H:/from_D/webdataset/datasets/soul_weights/soul_weights_dataset.jsonl", 20),
    ]

    all_ok = True
    for file_path, min_records in jsonl_files:
        path = Path(file_path)
        if path.exists():
            try:
                # Count lines to verify it's proper JSONL
                with open(path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)

                # Try to parse first line
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                    data = json.loads(first_line)
                    is_valid = isinstance(data, dict)

                if is_valid and line_count >= min_records:
                    print_status(True, f"{path.name}: {line_count} records")
                else:
                    print_status(
                        False,
                        f"{path.name}",
                        f"Invalid format or too few records ({line_count})",
                    )
                    all_ok = False
            except Exception as e:
                print_status(False, f"{path.name}", str(e))
                all_ok = False
        else:
            print_status(False, f"{path.name}", "File not found")
            all_ok = False

    return all_ok


def verify_ollama():
    """Verify Ollama models"""
    print_header("Phase 2: Ollama Configuration")

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            models = result.stdout

            # Check for Borea model
            if "borea-phi3.5" in models.lower():
                print_status(True, "borea-phi3.5-instruct-jp model found")
                borea_ok = True
            else:
                print_status(False, "borea-phi3.5-instruct-jp", "Model not in Ollama")
                borea_ok = False

            # Check for CPU Modelfile
            modelfile_path = Path("src/core/so8t/config/modelfiles/Modelfile-Borea-CPU")
            if modelfile_path.exists():
                print_status(True, f"Modelfile-Borea-CPU exists")
                modelfile_ok = True
            else:
                print_status(False, "Modelfile-Borea-CPU", "File not found")
                modelfile_ok = False

            return borea_ok and modelfile_ok
        else:
            print_status(False, "Ollama check", f"Command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print_status(False, "Ollama", "Ollama not installed or not in PATH")
        return False
    except Exception as e:
        print_status(False, "Ollama check", str(e))
        return False


def verify_datasets():
    """Verify Moonshot datasets exist"""
    print_header("Phase 3: Moonshot Datasets")

    project_root = Path(__file__).resolve().parents[1]
    moonshot_dir = project_root / "data" / "moonshot"

    required_datasets = [
        ("domain_knowledge.jsonl", 1000),
        ("arxiv_papers.jsonl", 100),
        ("nsfw_filtered.jsonl", 50),
        ("nsfw_detection.jsonl", 50),
        ("mcp_skills_integration.jsonl", 100),
        ("quadrality_allow_escalate_deny_refuse.jsonl", 100),
    ]

    all_ok = True
    for dataset, min_records in required_datasets:
        path = moonshot_dir / dataset
        if path.exists():
            try:
                # Count records
                with open(path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)

                if line_count >= min_records:
                    size_kb = path.stat().st_size / 1024
                    print_status(
                        True, f"{dataset}: {line_count} records ({size_kb:.1f} KB)"
                    )
                else:
                    print_status(False, dataset, f"Too few records: {line_count}")
                    all_ok = False
            except Exception as e:
                print_status(False, dataset, str(e))
                all_ok = False
        else:
            print_status(False, dataset, "File not found")
            all_ok = False

    return all_ok


def verify_imports():
    """Verify RTX3060DatasetPipeline can be imported"""
    print_header("Phase 4: RTX3060DatasetPipeline Import")

    project_root = Path(__file__).resolve().parents[1]

    # Add paths
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    try:
        # Try primary import
        try:
            from src.data.processing.dataset_pipeline import RTX3060DatasetPipeline

            print_status(
                True, "Primary import: from src.data.processing.dataset_pipeline"
            )
            primary_ok = True
        except ImportError as e:
            print_status(False, "Primary import failed", str(e))
            primary_ok = False

        # Try fallback import
        try:
            from data.processing.dataset_pipeline import RTX3060DatasetPipeline

            print_status(True, "Fallback import: from data.processing.dataset_pipeline")
            fallback_ok = True
        except ImportError as e:
            print_status(False, "Fallback import failed", str(e))
            fallback_ok = False

        # Try instantiation
        if primary_ok or fallback_ok:
            try:
                if primary_ok:
                    from src.data.processing.dataset_pipeline import (
                        RTX3060DatasetPipeline,
                    )
                else:
                    from data.processing.dataset_pipeline import RTX3060DatasetPipeline

                pipeline = RTX3060DatasetPipeline()
                print_status(True, "RTX3060DatasetPipeline instantiation")
                instantiate_ok = True
            except Exception as e:
                print_status(False, "Instantiation failed", str(e))
                instantiate_ok = False
        else:
            instantiate_ok = False

        return primary_ok or (fallback_ok and instantiate_ok)

    except Exception as e:
        print_status(False, "Import check", str(e))
        return False


def verify_scripts():
    """Verify helper scripts exist"""
    print_header("Phase 5: Helper Scripts")

    project_root = Path(__file__).resolve().parents[1]

    scripts = [
        ("scripts/fix_jsonl_schema.py", "JSONL schema fix script"),
        ("scripts/download_missing_datasets.py", "Dataset downloader script"),
        ("scripts/verify_pipeline_fixes.py", "This verification script"),
    ]

    all_ok = True
    for script_path, description in scripts:
        path = project_root / script_path
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print_status(True, f"{description}: {size_kb:.1f} KB")
        else:
            print_status(False, description, "File not found")
            all_ok = False

    return all_ok


def verify_training_config():
    """Verify training configuration"""
    print_header("Phase 6: Training Configuration")

    project_root = Path(__file__).resolve().parents[1]

    configs_to_check = [
        ("src/infrastructure/config/borea_training.json", "Borea training config"),
        ("src/infrastructure/config/dataset.json", "Dataset config"),
    ]

    all_ok = True
    for config_path, description in configs_to_check:
        path = project_root / config_path
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                print_status(True, f"{description}: Valid JSON")
            except json.JSONDecodeError as e:
                print_status(False, description, f"Invalid JSON: {e}")
                all_ok = False
            except Exception as e:
                print_status(False, description, str(e))
                all_ok = False
        else:
            print_status(False, description, "File not found")
            all_ok = False

    return all_ok


def main():
    """Main verification routine"""
    print("\n" + "=" * 70)
    print("  AEGIS PIPELINE STABILIZATION - VERIFICATION REPORT")
    print("=" * 70)
    print(
        f"  Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 70)

    results = {
        "JSONL Fixes": verify_jsonl_fixes(),
        "Ollama Config": verify_ollama(),
        "Datasets": verify_datasets(),
        "Imports": verify_imports(),
        "Scripts": verify_scripts(),
        "Training Config": verify_training_config(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[OK]" if passed else "[NG]"
        print(f"  {symbol} {check}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("  [OK] ALL CHECKS PASSED - Pipeline is ready!")
        print("=" * 70)
        print("\n  Next steps:")
        print("    1. Run: py -3 src/training/train_unsloth_so8t.py --max_steps 1")
        print("    2. Monitor logs for any remaining issues")
        print(
            "    3. Start full pipeline: .\\scripts\\pipeline\\run_aegis_continuous.ps1"
        )
        return 0
    else:
        passed_count = sum(results.values())
        total_count = len(results)
        print(f"  [NG] SOME CHECKS FAILED ({passed_count}/{total_count})")
        print("=" * 70)
        print("\n  Please review failed checks above and apply fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
