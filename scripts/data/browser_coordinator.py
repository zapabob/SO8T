#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ブラウザ間協調通信

MCPサーバーを介してブラウザ間で協調動作し、キーワードレベルの重複を回避する機能を実装。
中央コーディネーターとMCPサーバーによる状態共有を提供します。

Usage:
    from scripts.data.browser_coordinator import BrowserCoordinator
    
    coordinator = BrowserCoordinator(browser_id=0, mcp_config={...})
    await coordinator.start()
    await coordinator.broadcast_keyword_assignment("Python", browser_id=0)
    await coordinator.stop()
"""

import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from collections import defaultdict

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

# MCPクライアントインポート
try:
    from scripts.utils.mcp_client import MCPClient
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[COORDINATOR] MCPClient not available")

# キーワードコーディネーターインポート
try:
    from scripts.utils.keyword_coordinator import KeywordCoordinator
    KEYWORD_COORDINATOR_AVAILABLE = True
except ImportError:
    KEYWORD_COORDINATOR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[COORDINATOR] KeywordCoordinator not available")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/browser_coordinator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BrowserCoordinator:
    """ブラウザ間協調通信コーディネーター"""
    
    def __init__(
        self,
        browser_id: int,
        keyword_coordinator: Optional[KeywordCoordinator] = None,
        mcp_config: Optional[Dict[str, Any]] = None,
        heartbeat_interval: float = 30.0,
        broadcast_channel: str = "browser_coordination"
    ):
        """
        初期化
        
        Args:
            browser_id: ブラウザID
            keyword_coordinator: キーワードコーディネーターインスタンス
            mcp_config: MCPサーバー設定
            heartbeat_interval: ハートビート間隔（秒）
            broadcast_channel: ブロードキャストチャンネル名
        """
        self.browser_id = browser_id
        self.heartbeat_interval = heartbeat_interval
        self.broadcast_channel = broadcast_channel
        
        # キーワードコーディネーター
        if keyword_coordinator is None and KEYWORD_COORDINATOR_AVAILABLE:
            try:
                self.keyword_coordinator = KeywordCoordinator()
            except Exception as e:
                logger.warning(f"[COORDINATOR] Failed to initialize KeywordCoordinator: {e}")
                self.keyword_coordinator = None
        else:
            self.keyword_coordinator = keyword_coordinator
        
        # MCPクライアント
        self.mcp_client: Optional[MCPClient] = None
        self.mcp_connected = False
        
        if mcp_config and MCP_CLIENT_AVAILABLE:
            try:
                self.mcp_client = MCPClient(
                    transport=mcp_config.get('transport', 'stdio'),
                    command=mcp_config.get('command', 'npx'),
                    args=mcp_config.get('args', ['-y', '@modelcontextprotocol/server-chrome-devtools']),
                    url=mcp_config.get('url'),
                    timeout=mcp_config.get('timeout', 30000)
                )
            except Exception as e:
                logger.warning(f"[COORDINATOR] Failed to initialize MCPClient: {e}")
                self.mcp_client = None
        
        # 状態管理
        self.browser_states: Dict[int, Dict[str, Any]] = {}  # 他のブラウザの状態
        self.assigned_keywords: Set[str] = set()  # このブラウザに割り当てられたキーワード
        self.processing_keywords: Set[str] = set()  # 処理中のキーワード
        
        # メッセージハンドラー
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # バックグラウンドタスク
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_listener_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("="*80)
        logger.info("Browser Coordinator Initialized")
        logger.info("="*80)
        logger.info(f"Browser ID: {self.browser_id}")
        logger.info(f"MCP Client: {self.mcp_client is not None}")
        logger.info(f"Keyword Coordinator: {self.keyword_coordinator is not None}")
    
    async def start(self) -> bool:
        """
        コーディネーターを開始
        
        Returns:
            success: 成功フラグ
        """
        try:
            # MCPサーバーに接続
            if self.mcp_client:
                self.mcp_connected = await self.mcp_client.connect()
                if not self.mcp_connected:
                    logger.warning("[COORDINATOR] Failed to connect to MCP server, continuing without MCP")
            
            # バックグラウンドタスクを開始
            self.running = True
            
            if self.mcp_connected:
                # ハートビートタスクを開始
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                logger.info("[COORDINATOR] Heartbeat task started")
            
            # 初期状態をブロードキャスト
            await self.broadcast_status_update()
            
            logger.info("[COORDINATOR] Coordinator started")
            return True
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to start: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def stop(self) -> bool:
        """
        コーディネーターを停止
        
        Returns:
            success: 成功フラグ
        """
        try:
            self.running = False
            
            # バックグラウンドタスクを停止
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            if self.message_listener_task:
                self.message_listener_task.cancel()
                try:
                    await self.message_listener_task
                except asyncio.CancelledError:
                    pass
            
            # MCPサーバーから切断
            if self.mcp_client and self.mcp_connected:
                await self.mcp_client.disconnect()
                self.mcp_connected = False
            
            logger.info("[COORDINATOR] Coordinator stopped")
            return True
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to stop: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """ハートビートループ"""
        while self.running:
            try:
                await self.broadcast_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[COORDINATOR] Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def broadcast_heartbeat(self) -> bool:
        """
        ハートビートをブロードキャスト
        
        Returns:
            success: 成功フラグ
        """
        if not self.mcp_connected:
            return False
        
        try:
            message = {
                'type': 'heartbeat',
                'browser_id': self.browser_id,
                'timestamp': datetime.now().isoformat(),
                'assigned_keywords': list(self.assigned_keywords),
                'processing_keywords': list(self.processing_keywords)
            }
            
            # MCPサーバー経由でブロードキャスト（実装はMCPサーバーの機能に依存）
            # 現在のMCP実装では直接ブロードキャスト機能がないため、
            # 共有メモリ（JSONファイル）を使用して状態を共有
            await self._save_browser_state(message)
            
            return True
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to broadcast heartbeat: {e}")
            return False
    
    async def broadcast_keyword_assignment(self, keyword: str) -> bool:
        """
        キーワード割り当てをブロードキャスト
        
        Args:
            keyword: 割り当てられたキーワード
        
        Returns:
            success: 成功フラグ
        """
        try:
            message = {
                'type': 'keyword_assignment',
                'browser_id': self.browser_id,
                'keyword': keyword,
                'timestamp': datetime.now().isoformat()
            }
            
            # 状態を更新
            self.assigned_keywords.add(keyword.lower())
            
            # ブロードキャスト
            await self._save_browser_state(message)
            await self.broadcast_status_update()
            
            logger.info(f"[COORDINATOR] Broadcasted keyword assignment: {keyword} (browser {self.browser_id})")
            return True
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to broadcast keyword assignment: {e}")
            return False
    
    async def broadcast_keyword_completion(self, keyword: str, success: bool = True) -> bool:
        """
        キーワード完了をブロードキャスト
        
        Args:
            keyword: 完了したキーワード
            success: 成功フラグ
        
        Returns:
            success: 成功フラグ
        """
        try:
            message = {
                'type': 'keyword_completion',
                'browser_id': self.browser_id,
                'keyword': keyword,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            # 状態を更新
            keyword_lower = keyword.lower()
            self.assigned_keywords.discard(keyword_lower)
            self.processing_keywords.discard(keyword_lower)
            
            # ブロードキャスト
            await self._save_browser_state(message)
            await self.broadcast_status_update()
            
            logger.info(f"[COORDINATOR] Broadcasted keyword completion: {keyword} (browser {self.browser_id}, success: {success})")
            return True
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to broadcast keyword completion: {e}")
            return False
    
    async def broadcast_status_update(self) -> bool:
        """
        状態更新をブロードキャスト
        
        Returns:
            success: 成功フラグ
        """
        try:
            message = {
                'type': 'status_update',
                'browser_id': self.browser_id,
                'timestamp': datetime.now().isoformat(),
                'assigned_keywords': list(self.assigned_keywords),
                'processing_keywords': list(self.processing_keywords),
                'status': 'active'
            }
            
            await self._save_browser_state(message)
            return True
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to broadcast status update: {e}")
            return False
    
    async def _save_browser_state(self, message: Dict[str, Any]) -> bool:
        """
        ブラウザ状態を共有メモリに保存
        
        Args:
            message: メッセージ
        
        Returns:
            success: 成功フラグ
        """
        try:
            # 共有メモリファイル（JSON）
            state_file = Path("D:/webdataset/checkpoints/browser_coordination_state.json")
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 既存の状態を読み込み
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            else:
                state = {
                    'browsers': {},
                    'messages': [],
                    'last_updated': datetime.now().isoformat()
                }
            
            # ブラウザ状態を更新
            browser_id = message.get('browser_id', self.browser_id)
            if 'browsers' not in state:
                state['browsers'] = {}
            
            state['browsers'][str(browser_id)] = {
                'browser_id': browser_id,
                'last_heartbeat': datetime.now().isoformat(),
                'assigned_keywords': message.get('assigned_keywords', []),
                'processing_keywords': message.get('processing_keywords', []),
                'status': message.get('status', 'active')
            }
            
            # メッセージを追加（最新100件のみ保持）
            if 'messages' not in state:
                state['messages'] = []
            state['messages'].append(message)
            state['messages'] = state['messages'][-100:]
            
            state['last_updated'] = datetime.now().isoformat()
            
            # 状態を保存
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to save browser state: {e}")
            return False
    
    async def get_other_browsers_keywords(self) -> Set[str]:
        """
        他のブラウザが処理中のキーワードを取得
        
        Returns:
            keywords: 他のブラウザが処理中のキーワードセット
        """
        try:
            state_file = Path("D:/webdataset/checkpoints/browser_coordination_state.json")
            
            if not state_file.exists():
                return set()
            
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            other_keywords = set()
            browsers = state.get('browsers', {})
            
            for browser_id_str, browser_state in browsers.items():
                browser_id = int(browser_id_str)
                if browser_id != self.browser_id:
                    # 他のブラウザの割り当て済みキーワードと処理中キーワードを取得
                    assigned = browser_state.get('assigned_keywords', [])
                    processing = browser_state.get('processing_keywords', [])
                    other_keywords.update([k.lower() for k in assigned + processing])
            
            return other_keywords
            
        except Exception as e:
            logger.error(f"[COORDINATOR] Failed to get other browsers keywords: {e}")
            return set()
    
    async def get_next_keyword_with_coordination(self) -> Optional[str]:
        """
        協調動作で次のキーワードを取得（重複回避）
        
        Returns:
            keyword: 割り当てられたキーワード（Noneの場合は利用可能なキーワードなし）
        """
        if not self.keyword_coordinator:
            logger.warning("[COORDINATOR] KeywordCoordinator not available")
            return None
        
        # 他のブラウザが処理中のキーワードを取得
        other_keywords = await self.get_other_browsers_keywords()
        
        # キーワードコーディネーターから次のキーワードを取得
        keyword = self.keyword_coordinator.get_next_keyword(self.browser_id)
        
        if keyword:
            keyword_lower = keyword.lower()
            
            # 他のブラウザが処理中でないことを確認
            if keyword_lower in other_keywords:
                logger.warning(f"[COORDINATOR] Keyword '{keyword}' is already being processed by another browser")
                # キーワードをpendingに戻す
                self.keyword_coordinator.mark_keyword_failed(keyword, self.browser_id, "Already assigned to another browser")
                return None
            
            # 割り当てをブロードキャスト
            await self.broadcast_keyword_assignment(keyword)
            
            return keyword
        
        return None
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """
        メッセージハンドラーを登録
        
        Args:
            message_type: メッセージタイプ
            handler: ハンドラー関数
        """
        self.message_handlers[message_type].append(handler)
        logger.debug(f"[COORDINATOR] Registered handler for message type: {message_type}")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """
        メッセージを処理
        
        Args:
            message: メッセージ
        """
        message_type = message.get('type')
        browser_id = message.get('browser_id')
        
        # 自分のメッセージは無視
        if browser_id == self.browser_id:
            return
        
        # ハンドラーを実行
        handlers = self.message_handlers.get(message_type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"[COORDINATOR] Handler error for message type {message_type}: {e}")
    
    def get_browser_states(self) -> Dict[int, Dict[str, Any]]:
        """
        すべてのブラウザの状態を取得
        
        Returns:
            states: ブラウザ状態辞書
        """
        return self.browser_states.copy()
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """
        協調動作の統計情報を取得
        
        Returns:
            stats: 統計情報辞書
        """
        stats = {
            'browser_id': self.browser_id,
            'assigned_keywords_count': len(self.assigned_keywords),
            'processing_keywords_count': len(self.processing_keywords),
            'other_browsers_count': len(self.browser_states),
            'mcp_connected': self.mcp_connected
        }
        
        return stats


async def main():
    """メイン関数（テスト用）"""
    coordinator = BrowserCoordinator(
        browser_id=0,
        mcp_config={
            'transport': 'stdio',
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-chrome-devtools']
        }
    )
    
    try:
        # コーディネーターを開始
        await coordinator.start()
        
        # キーワードを取得
        keyword = await coordinator.get_next_keyword_with_coordination()
        print(f"Got keyword: {keyword}")
        
        # 完了をブロードキャスト
        if keyword:
            await coordinator.broadcast_keyword_completion(keyword, success=True)
        
        # 少し待機
        await asyncio.sleep(5)
        
        # コーディネーターを停止
        await coordinator.stop()
        
    except Exception as e:
        logger.error(f"[COORDINATOR] Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await coordinator.stop()


if __name__ == "__main__":
    asyncio.run(main())














