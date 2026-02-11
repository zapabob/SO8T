"""
Minimal AGI loop tying together SQL audit log, perception, and SO8T inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from perception.opencv_ingest import summarize_video
from safety_sql.sqlmm import SQLMemoryManager
from scripts.demo_infer import infer


def run_loop(database: Path, text: str, video: Optional[Path]) -> None:
    manager = SQLMemoryManager(database)
    decision_output = infer(text, checkpoint=None)
    decision = decision_output["decision"]
    manager.log_decision(
        conversation_id="demo",
        user_input=text,
        model_output=str(decision_output),
        decision=decision,
        verifier_score=float(decision_output["score"]),
    )

    if video:
        for summary in summarize_video(video, stride=30):
            manager.log_audit_event("vision_summary", summary.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    run_loop(args.database, args.text, args.video)


if __name__ == "__main__":
    main()
