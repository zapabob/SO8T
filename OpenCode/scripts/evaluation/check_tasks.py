#!/usr/bin/env python3
"""Check available lm_eval tasks"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lm-evaluation-harness'))

import lm_eval

def main():
    tasks = lm_eval.list_tasks()

    hellaswag_tasks = [t for t in tasks if 'hellaswag' in t.lower()]
    mmlu_tasks = [t for t in tasks if 'mmlu' in t.lower()]
    elyza_tasks = [t for t in tasks if 'elyza' in t.lower()]

    print("Available tasks sample:")
    print(f"Hellaswag tasks: {hellaswag_tasks[:5]}")
    print(f"MMLU tasks: {mmlu_tasks[:5]}")
    print(f"ELYZA tasks: {elyza_tasks[:5]}")

    # Check if specific tasks exist
    target_tasks = ['hellaswag', 'mmlu', 'elyza_tasks']
    available_targets = []

    for task in target_tasks:
        if task in tasks:
            available_targets.append(task)
        else:
            # Try partial matches
            matches = [t for t in tasks if task in t.lower()]
            if matches:
                available_targets.extend(matches[:3])

    print(f"\nTarget tasks available: {available_targets}")

if __name__ == "__main__":
    main()