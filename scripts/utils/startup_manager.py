import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StartupManager:
    """Windows スタートアップフォルダへの登録・削除を管理するクラス"""
    
    def __init__(self, script_path: Path):
        self.script_path = script_path.absolute()
        self.startup_folder = Path(os.getenv('APPDATA')) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        self.bat_name = "run_moonshot_pipeline_auto_resume.bat"
        self.bat_path = self.startup_folder / self.bat_name
        
    def register(self):
        """スタートアップに .bat ファイルを作成して登録"""
        try:
            if not self.startup_folder.exists():
                logger.warning(f"Startup folder not found: {self.startup_folder}")
                return False
                
            # .bat ファイルの内容を作成
            # py -3 を使用し、スクリプトがあるディレクトリに cd してから実行
            project_root = self.script_path.parent
            bat_content = f"""@echo off
cd /d "{project_root}"
py -3 "{self.script_path.name}" --use-existing-datasets
"""
            
            with open(self.bat_path, "w", encoding="cp932") as f:
                f.write(bat_content)
                
            logger.info(f"✅ スタートアップに登録完了: {self.bat_path}")
            return True
        except Exception as e:
            logger.error(f"❌ スタートアップ登録失敗: {e}")
            return False
            
    def unregister(self):
        """スタートアップから .bat ファイルを削除"""
        try:
            if self.bat_path.exists():
                os.remove(self.bat_path)
                logger.info(f"🗑️ スタートアップから削除完了: {self.bat_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ スタートアップ削除失敗: {e}")
            return False

    def is_registered(self) -> bool:
        """登録済みか確認"""
        return self.bat_path.exists()
