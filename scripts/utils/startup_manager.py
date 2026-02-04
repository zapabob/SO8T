import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class StartupManager:
    """Windows 繧ｹ繧ｿ繝ｼ繝医い繝・・縺ｫ閾ｪ蜍募・髢狗畑 .bat 繧堤匳骭ｲ/隗｣髯､縺吶ｋ縲・"""

    def __init__(self, script_path: Path):
        self.script_path = script_path.absolute()
        self.startup_folder = (
            Path(os.getenv("APPDATA"))
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            / "Startup"
        )
        self.bat_name = "run_moonshot_pipeline_auto_resume.bat"
        self.bat_path = self.startup_folder / self.bat_name

    def register(self, extra_args: Optional[List[str]] = None) -> bool:
        """繧ｹ繧ｿ繝ｼ繝医い繝・・縺ｫ .bat 繧堤匳骭ｲ縺吶ｋ縲・"""
        try:
            if not self.startup_folder.exists():
                logger.warning("Startup folder not found: %s", self.startup_folder)
                return False

            project_root = self.script_path.parent
            extra_args = extra_args or []

            def _quote(arg: str) -> str:
                if " " in arg or "\t" in arg or "\"" in arg:
                    return f'"{arg.replace("\"", "\\\"")}"'
                return arg

            arg_string = " ".join(_quote(arg) for arg in extra_args)

            env_lines = [
                "set SO8T_CHECKPOINT_INTERVAL=300",
                "set SO8T_ROLLING_CHECKPOINTS=5",
            ]
            optional_envs = [
                "SO8T_GRAPE_VARIANT",
                "SO8T_USE_UNSLOTH",
                "SO8T_MCP_API_SKILL",
                "SO8T_RECOVER",
                "SO8T_TRAINING_CONFIG",
                "SO8T_SUBAGENT_STRATEGY",
                "SO8T_SUBAGENT_SCHEDULE",
                "SO8T_MHC_ENABLE",
                "SO8T_MHC_TARGETS",
                "SO8T_MHC_BLEND",
                "SO8T_SO8_ENABLE",
                "SO8T_SO8_MODE",
            ]
            for key in optional_envs:
                value = os.getenv(key)
                if value:
                    env_lines.append(f'set "{key}={value}"')

            env_block = "\n".join(env_lines)

            bat_content = f"""@echo off
cd /d "{project_root}"
{env_block}
start /B py -3 "{self.script_path.name}" {arg_string} > nul 2>&1
timeout /t 5 > nul
"""

            with open(self.bat_path, "w", encoding="cp932") as f:
                f.write(bat_content)

            logger.info("Startup registered: %s", self.bat_path)
            return True
        except Exception as exc:
            logger.error("Startup registration failed: %s", exc)
            return False

    def unregister(self) -> bool:
        """繧ｹ繧ｿ繝ｼ繝医い繝・・縺九ｉ .bat 繧貞炎髯､縺吶ｋ縲・"""
        try:
            if self.bat_path.exists():
                self.bat_path.unlink()
                logger.info("Startup unregistered: %s", self.bat_path)
                return True
            return False
        except Exception as exc:
            logger.error("Startup unregister failed: %s", exc)
            return False

    def is_registered(self) -> bool:
        return self.bat_path.exists()
