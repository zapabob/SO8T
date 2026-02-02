#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Chrome DevTools実際起動機能

Chrome DevToolsを実際に開き、MCP Chrome DevToolsサーバーに接続して
複数のインスタンスを管理する機能を提供します。

Usage:
    from scripts.utils.chrome_devtools_launcher import ChromeDevToolsLauncher
    
    launcher = ChromeDevToolsLauncher()
    await launcher.start_server(instance_id=0)
    mcp_wrapper = await launcher.get_mcp_wrapper(instance_id=0)
"""

import sys
import logging
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class ChromeDevToolsLauncher:
    """Chrome DevTools実際起動クラス"""
    
    def __init__(
        self,
        transport: str = "stdio",
        command: str = "npx",
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        timeout: int = 30000,
        max_instances: int = 10
    ):
        """
        初期化
        
        Args:
            transport: トランスポートタイプ（stdio, http, websocket）
            command: MCPサーバー起動コマンド（stdioの場合）
            args: MCPサーバー起動引数（stdioの場合）
            url: MCPサーバーURL（http/websocketの場合）
            timeout: タイムアウト（ミリ秒）
            max_instances: 最大インスタンス数
        """
        self.transport = transport
        self.command = command
        self.args = args or ["-y", "@modelcontextprotocol/server-chrome-devtools"]
        self.url = url
        self.timeout = timeout
        self.max_instances = max_instances
        
        # インスタンス管理
        self.instances: Dict[int, Dict[str, Any]] = {}
        self.mcp_wrappers: Dict[int, Any] = {}
        
        # プロセス管理
        self.processes: Dict[int, subprocess.Popen] = {}
        
        logger.info("="*80)
        logger.info("Chrome DevTools Launcher Initialized")
        logger.info("="*80)
        logger.info(f"Transport: {self.transport}")
        logger.info(f"Command: {self.command}")
        logger.info(f"Args: {self.args}")
        logger.info(f"Max instances: {self.max_instances}")
    
    async def start_server(self, instance_id: int = 0) -> bool:
        """
        Chrome DevToolsサーバーを起動
        
        Args:
            instance_id: インスタンスID（0-9）
        
        Returns:
            success: 成功フラグ
        """
        if instance_id < 0 or instance_id >= self.max_instances:
            logger.error(f"[INSTANCE {instance_id}] Invalid instance ID (must be 0-{self.max_instances-1})")
            return False
        
        if instance_id in self.instances:
            logger.warning(f"[INSTANCE {instance_id}] Already started")
            return True
        
        try:
            logger.info(f"[INSTANCE {instance_id}] Starting Chrome DevTools server...")
            
            # MCP Chrome DevToolsラッパーを初期化
            from scripts.utils.mcp_chrome_devtools_wrapper import MCPChromeDevTools
            
            mcp_wrapper = MCPChromeDevTools(
                transport=self.transport,
                command=self.command,
                args=self.args,
                url=self.url,
                timeout=self.timeout
            )
            
            # MCPサーバーに接続
            connected = await mcp_wrapper.connect()
            
            if not connected:
                logger.error(f"[INSTANCE {instance_id}] Failed to connect to MCP server")
                return False
            
            # インスタンス情報を保存
            self.instances[instance_id] = {
                'instance_id': instance_id,
                'mcp_wrapper': mcp_wrapper,
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'transport': self.transport,
                'command': self.command,
                'args': self.args
            }
            
            self.mcp_wrappers[instance_id] = mcp_wrapper
            
            logger.info(f"[OK] Instance {instance_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"[INSTANCE {instance_id}] Failed to start: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def stop_server(self, instance_id: int) -> bool:
        """
        Chrome DevToolsサーバーを停止
        
        Args:
            instance_id: インスタンスID
        
        Returns:
            success: 成功フラグ
        """
        if instance_id not in self.instances:
            logger.warning(f"[INSTANCE {instance_id}] Not started")
            return False
        
        try:
            logger.info(f"[INSTANCE {instance_id}] Stopping Chrome DevTools server...")
            
            # MCPラッパーから切断
            if instance_id in self.mcp_wrappers:
                mcp_wrapper = self.mcp_wrappers[instance_id]
                await mcp_wrapper.disconnect()
                del self.mcp_wrappers[instance_id]
            
            # プロセスを終了
            if instance_id in self.processes:
                process = self.processes[instance_id]
                try:
                    process.terminate()
                    await asyncio.sleep(1)
                    if process.poll() is None:
                        process.kill()
                except Exception as e:
                    logger.warning(f"[INSTANCE {instance_id}] Failed to terminate process: {e}")
                del self.processes[instance_id]
            
            # インスタンス情報を削除
            self.instances[instance_id]['status'] = 'stopped'
            self.instances[instance_id]['stopped_at'] = datetime.now().isoformat()
            del self.instances[instance_id]
            
            logger.info(f"[OK] Instance {instance_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"[INSTANCE {instance_id}] Failed to stop: {e}")
            return False
    
    async def get_mcp_wrapper(self, instance_id: int) -> Optional[Any]:
        """
        MCPラッパーを取得
        
        Args:
            instance_id: インスタンスID
        
        Returns:
            mcp_wrapper: MCPラッパーインスタンス
        """
        if instance_id not in self.instances:
            logger.warning(f"[INSTANCE {instance_id}] Not started, starting now...")
            success = await self.start_server(instance_id)
            if not success:
                return None
        
        return self.mcp_wrappers.get(instance_id)
    
    async def start_all_instances(self) -> bool:
        """
        すべてのインスタンスを起動
        
        Returns:
            success: 成功フラグ
        """
        logger.info(f"[START] Starting all {self.max_instances} instances...")
        
        success_count = 0
        for instance_id in range(self.max_instances):
            success = await self.start_server(instance_id)
            if success:
                success_count += 1
            # インスタンス間で少し待機（リソース競合を避ける）
            await asyncio.sleep(0.5)
        
        logger.info(f"[OK] Started {success_count}/{self.max_instances} instances")
        return success_count == self.max_instances
    
    async def stop_all_instances(self) -> bool:
        """
        すべてのインスタンスを停止
        
        Returns:
            success: 成功フラグ
        """
        logger.info(f"[STOP] Stopping all instances...")
        
        success_count = 0
        for instance_id in list(self.instances.keys()):
            success = await self.stop_server(instance_id)
            if success:
                success_count += 1
        
        logger.info(f"[OK] Stopped {success_count} instances")
        return True
    
    def get_instance_status(self, instance_id: int) -> Optional[Dict[str, Any]]:
        """
        インスタンスの状態を取得
        
        Args:
            instance_id: インスタンスID
        
        Returns:
            status: インスタンス状態
        """
        return self.instances.get(instance_id)
    
    def get_all_instances_status(self) -> Dict[int, Dict[str, Any]]:
        """
        すべてのインスタンスの状態を取得
        
        Returns:
            statuses: すべてのインスタンス状態
        """
        return self.instances.copy()
    
    async def restart_server(self, instance_id: int) -> bool:
        """
        サーバーを再起動
        
        Args:
            instance_id: インスタンスID
        
        Returns:
            success: 成功フラグ
        """
        logger.info(f"[INSTANCE {instance_id}] Restarting server...")
        
        # 停止
        await self.stop_server(instance_id)
        
        # 少し待機
        await asyncio.sleep(1)
        
        # 起動
        return await self.start_server(instance_id)


async def main():
    """メイン関数（テスト用）"""
    launcher = ChromeDevToolsLauncher(max_instances=10)
    
    try:
        # すべてのインスタンスを起動
        await launcher.start_all_instances()
        
        # 状態を確認
        statuses = launcher.get_all_instances_status()
        logger.info(f"[STATUS] Active instances: {len(statuses)}")
        
        # 10秒待機
        await asyncio.sleep(10)
        
        # すべてのインスタンスを停止
        await launcher.stop_all_instances()
        
    except KeyboardInterrupt:
        logger.warning("[INTERRUPT] Interrupted by user")
        await launcher.stop_all_instances()
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await launcher.stop_all_instances()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())

