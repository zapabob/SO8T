"""実装ログ作成スクリプト"""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

def get_worktree_name():
    """現在のgit worktree名を取得"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True
        )
        git_dir = result.stdout.strip()
        
        if "worktrees" in git_dir:
            parts = Path(git_dir).parts
            if "worktrees" in parts:
                idx = parts.index("worktrees")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        return "main"
    except Exception:
        return "main"

if __name__ == "__main__":
    worktree = get_worktree_name()
    today = datetime.now().strftime("%Y-%m-%d")
    feature_name = sys.argv[1] if len(sys.argv) > 1 else "feature"
    filename = f"{today}_{worktree}_{feature_name}.md"
    print(filename)



















































