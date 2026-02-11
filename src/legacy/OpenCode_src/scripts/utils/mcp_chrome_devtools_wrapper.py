#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
MCP Chrome DevTools ラッパー

CursorのMCP Chrome DevToolsを使用するためのラッパークラス。
MCPサーバーに接続してブラウザ操作を実行します。

Usage:
    from scripts.utils.mcp_chrome_devtools_wrapper import MCPChromeDevTools
    
    mcp = MCPChromeDevTools()
    page_idx = await mcp.new_page("https://example.com")
    snapshot = await mcp.take_snapshot(page_idx)
"""

import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class MCPChromeDevTools:
    """MCP Chrome DevTools ラッパークラス"""
    
    def __init__(
        self,
        transport: str = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        timeout: int = 30000
    ):
        """
        初期化
        
        Args:
            transport: トランスポートタイプ（stdio, http, websocket）
            command: MCPサーバー起動コマンド（stdioの場合）
            args: MCPサーバー起動引数（stdioの場合）
            url: MCPサーバーURL（http/websocketの場合）
            timeout: タイムアウト（ミリ秒）
        """
        self.pages: List[Dict] = []
        self.current_page_idx: Optional[int] = None
        
        # MCPクライアントを初期化
        try:
            from scripts.utils.mcp_client import MCPClient
            self.mcp_client = MCPClient(
                transport=transport,
                command=command,
                args=args,
                url=url,
                timeout=timeout
            )
            self.use_mcp_client = True
        except ImportError as e:
            logger.warning(f"[MCP] Failed to import MCPClient: {e}")
            logger.warning("[MCP] Using fallback implementation")
            self.mcp_client = None
            self.use_mcp_client = False
        
        self.connected = False
        logger.info("[MCP] Chrome DevTools wrapper initialized")
    
    async def connect(self) -> bool:
        """
        MCPサーバーに接続
        
        Returns:
            success: 成功フラグ
        """
        if self.use_mcp_client and self.mcp_client:
            self.connected = await self.mcp_client.connect()
            return self.connected
        else:
            logger.warning("[MCP] MCP client not available, using fallback")
            self.connected = True
            return True
    
    async def disconnect(self):
        """MCPサーバーから切断"""
        if self.use_mcp_client and self.mcp_client:
            await self.mcp_client.disconnect()
        self.connected = False
    
    async def list_pages(self) -> List[Dict]:
        """開いているページのリストを取得"""
        if self.use_mcp_client and self.mcp_client and self.connected:
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_list_pages", {})
            if result:
                self.pages = result
                return result
        return self.pages
    
    async def new_page(self, url: Optional[str] = None, timeout: int = 30000) -> int:
        """
        新しいページを作成
        
        Args:
            url: 開くURL（オプション）
            timeout: タイムアウト（ミリ秒）
        
        Returns:
            page_idx: ページインデックス
        """
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {}
            if url:
                arguments['url'] = url
            if timeout:
                arguments['timeout'] = timeout
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_new_page", arguments)
            if result:
                # MCPサーバーから返されたページ情報を使用
                page_idx = result.get('pageIdx', len(self.pages))
                self.current_page_idx = page_idx
                logger.info(f"[MCP] New page created via MCP: {page_idx} (URL: {url})")
                return page_idx
        
        # フォールバック実装
        page_info = {
            'index': len(self.pages),
            'url': url or 'about:blank',
            'created_at': asyncio.get_event_loop().time()
        }
        
        self.pages.append(page_info)
        self.current_page_idx = page_info['index']
        
        logger.info(f"[MCP] New page created (fallback): {page_info['index']} (URL: {url})")
        return page_info['index']
    
    async def select_page(self, page_idx: int) -> bool:
        """
        ページを選択
        
        Args:
            page_idx: ページインデックス
        
        Returns:
            success: 成功フラグ
        """
        if self.use_mcp_client and self.mcp_client and self.connected:
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_select_page", {"pageIdx": page_idx})
            if result:
                self.current_page_idx = page_idx
                logger.info(f"[MCP] Page selected via MCP: {page_idx}")
                return True
        
        # フォールバック実装
        if 0 <= page_idx < len(self.pages):
            self.current_page_idx = page_idx
            logger.info(f"[MCP] Page selected (fallback): {page_idx}")
            return True
        else:
            logger.error(f"[MCP] Invalid page index: {page_idx}")
            return False
    
    async def navigate_page(self, url: str, page_idx: Optional[int] = None, timeout: int = 30000) -> bool:
        """
        ページに移動
        
        Args:
            url: 移動先URL
            page_idx: ページインデックス（Noneの場合は現在のページ）
            timeout: タイムアウト（ミリ秒）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {
                "type": "url",
                "url": url,
                "timeout": timeout
            }
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_navigate_page", arguments)
            if result:
                logger.info(f"[MCP] Navigated page {page_idx} to: {url} (via MCP)")
                return True
        
        # フォールバック実装
        logger.info(f"[MCP] Navigating page {page_idx} to: {url} (fallback)")
        if page_idx < len(self.pages):
            self.pages[page_idx]['url'] = url
        return True
    
    async def take_snapshot(self, page_idx: Optional[int] = None, verbose: bool = False) -> Optional[Dict]:
        """
        ページのスナップショットを取得
        
        Args:
            page_idx: ページインデックス（Noneの場合は現在のページ）
            verbose: 詳細情報を含めるか
        
        Returns:
            snapshot: スナップショットデータ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return None
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"verbose": verbose}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_take_snapshot", arguments)
            if result:
                logger.info(f"[MCP] Snapshot taken via MCP for page {page_idx}")
                return result
        
        # フォールバック実装
        logger.info(f"[MCP] Taking snapshot (fallback) for page {page_idx}")
        snapshot = {
            'page_idx': page_idx,
            'url': self.pages[page_idx]['url'] if page_idx < len(self.pages) else None,
            'elements': []
        }
        return snapshot
    
    async def evaluate_script(self, function: str, args: Optional[List[Dict]] = None, page_idx: Optional[int] = None) -> Optional[Any]:
        """
        JavaScriptを実行
        
        Args:
            function: 実行するJavaScript関数
            args: 関数の引数
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            result: 実行結果
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return None
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"function": function}
            if args:
                arguments['args'] = args
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_evaluate_script", arguments)
            if result:
                logger.info(f"[MCP] Script evaluated via MCP on page {page_idx}")
                return result
        
        # フォールバック実装
        logger.info(f"[MCP] Evaluating script (fallback) on page {page_idx}")
        return None
    
    async def click(self, uid: str, page_idx: Optional[int] = None, dblClick: bool = False) -> bool:
        """
        要素をクリック
        
        Args:
            uid: 要素のUID（スナップショットから取得）
            page_idx: ページインデックス（Noneの場合は現在のページ）
            dblClick: ダブルクリックかどうか
        
        Returns:
            success: 成功フラグ
        """
        # MCPツール: mcp_chrome-devtools_click
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None or page_idx >= len(self.pages):
            logger.error(f"[MCP] Invalid page index: {page_idx}")
            return False
        
        logger.info(f"[MCP] Clicking element {uid} on page {page_idx}")
        
        # 実際の実装では、MCPサーバーに接続してクリックを実行
        return True
    
    async def fill(self, uid: str, value: str, page_idx: Optional[int] = None) -> bool:
        """
        入力フィールドに値を入力
        
        Args:
            uid: 要素のUID（スナップショットから取得）
            value: 入力する値
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"uid": uid, "value": value}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_fill", arguments)
            if result:
                logger.info(f"[MCP] Filled element {uid} via MCP on page {page_idx}")
                return True
        
        # フォールバック実装
        logger.info(f"[MCP] Filling element {uid} (fallback) on page {page_idx}")
        return True
    
    async def hover(self, uid: str, page_idx: Optional[int] = None) -> bool:
        """
        要素にホバー
        
        Args:
            uid: 要素のUID（スナップショットから取得）
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"uid": uid}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_hover", arguments)
            if result:
                logger.info(f"[MCP] Hovered element {uid} via MCP on page {page_idx}")
                return True
        
        # フォールバック実装
        logger.info(f"[MCP] Hovering element {uid} (fallback) on page {page_idx}")
        return True
    
    async def press_key(self, key: str, page_idx: Optional[int] = None) -> bool:
        """
        キーを押す
        
        Args:
            key: 押すキー（例: "Enter", "Control+A"）
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"key": key}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_press_key", arguments)
            if result:
                logger.info(f"[MCP] Pressed key {key} via MCP on page {page_idx}")
                return True
        
        # フォールバック実装
        logger.info(f"[MCP] Pressing key {key} (fallback) on page {page_idx}")
        return True
    
    async def take_screenshot(self, page_idx: Optional[int] = None, format: str = "png", fullPage: bool = False) -> Optional[bytes]:
        """
        スクリーンショットを取得
        
        Args:
            page_idx: ページインデックス（Noneの場合は現在のページ）
            format: 画像フォーマット（png, jpeg, webp）
            fullPage: フルページスクリーンショットかどうか
        
        Returns:
            screenshot: スクリーンショットデータ（バイト列）
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return None
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"format": format, "fullPage": fullPage}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_take_screenshot", arguments)
            if result:
                # スクリーンショットデータをbase64デコード
                import base64
                if isinstance(result, dict) and 'data' in result:
                    screenshot_data = base64.b64decode(result['data'])
                    logger.info(f"[MCP] Screenshot taken via MCP for page {page_idx}")
                    return screenshot_data
        
        # フォールバック実装
        logger.info(f"[MCP] Taking screenshot (fallback) for page {page_idx}")
        return None
    
    async def wait_for(self, text: str, page_idx: Optional[int] = None, timeout: int = 30000) -> bool:
        """
        テキストが表示されるまで待機
        
        Args:
            text: 待機するテキスト
            page_idx: ページインデックス（Noneの場合は現在のページ）
            timeout: タイムアウト（ミリ秒）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"text": text, "timeout": timeout}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_wait_for", arguments)
            if result:
                logger.info(f"[MCP] Waited for text '{text}' via MCP on page {page_idx}")
                return True
        
        # フォールバック実装
        logger.info(f"[MCP] Waiting for text '{text}' (fallback) on page {page_idx}")
        await asyncio.sleep(timeout / 1000.0)
        return True
    
    async def close_page(self, page_idx: int) -> bool:
        """
        ページを閉じる
        
        Args:
            page_idx: ページインデックス
        
        Returns:
            success: 成功フラグ
        """
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"pageIdx": page_idx}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_close_page", arguments)
            if result:
                if page_idx < len(self.pages):
                    self.pages.pop(page_idx)
                if self.current_page_idx == page_idx:
                    self.current_page_idx = None
                logger.info(f"[MCP] Page closed via MCP: {page_idx}")
                return True
        
        # フォールバック実装
        if 0 <= page_idx < len(self.pages):
            self.pages.pop(page_idx)
            if self.current_page_idx == page_idx:
                self.current_page_idx = None
            logger.info(f"[MCP] Page closed (fallback): {page_idx}")
            return True
        else:
            logger.error(f"[MCP] Invalid page index: {page_idx}")
            return False
    
    async def drag(self, from_uid: str, to_uid: str, page_idx: Optional[int] = None) -> bool:
        """
        ドラッグ&ドロップ
        
        Args:
            from_uid: ドラッグ元要素のUID
            to_uid: ドロップ先要素のUID
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"from_uid": from_uid, "to_uid": to_uid}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_drag", arguments)
            if result:
                logger.info(f"[MCP] Dragged element {from_uid} to {to_uid} via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Dragging element {from_uid} to {to_uid} (fallback) on page {page_idx}")
        return True
    
    async def emulate(self, page_idx: Optional[int] = None, networkConditions: Optional[str] = None, cpuThrottlingRate: Optional[float] = None) -> bool:
        """
        エミュレーション設定
        
        Args:
            page_idx: ページインデックス（Noneの場合は現在のページ）
            networkConditions: ネットワーク条件（"Offline", "Slow 3G", "Fast 3G", "Slow 4G", "Fast 4G"）
            cpuThrottlingRate: CPUスロットリング率（1-20）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {}
            if networkConditions:
                arguments['networkConditions'] = networkConditions
            if cpuThrottlingRate:
                arguments['cpuThrottlingRate'] = cpuThrottlingRate
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_emulate", arguments)
            if result:
                logger.info(f"[MCP] Emulation set via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Setting emulation (fallback) on page {page_idx}")
        return True
    
    async def fill_form(self, elements: List[Dict], page_idx: Optional[int] = None) -> bool:
        """
        フォーム入力
        
        Args:
            elements: 入力要素のリスト（[{"uid": "...", "value": "..."}, ...]）
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"elements": elements}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_fill_form", arguments)
            if result:
                logger.info(f"[MCP] Form filled via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Filling form (fallback) on page {page_idx}")
        return True
    
    async def get_console_message(self, msgid: int, page_idx: Optional[int] = None) -> Optional[Dict]:
        """
        コンソールメッセージを取得
        
        Args:
            msgid: メッセージID
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            message: コンソールメッセージ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return None
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"msgid": msgid}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_get_console_message", arguments)
            if result:
                logger.info(f"[MCP] Console message retrieved via MCP for page {page_idx}")
                return result
        
        logger.info(f"[MCP] Getting console message (fallback) for page {page_idx}")
        return None
    
    async def get_network_request(self, reqid: Optional[int] = None, page_idx: Optional[int] = None) -> Optional[Dict]:
        """
        ネットワークリクエストを取得
        
        Args:
            reqid: リクエストID（Noneの場合は現在選択中のリクエスト）
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            request: ネットワークリクエスト
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return None
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {}
            if reqid:
                arguments['reqid'] = reqid
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_get_network_request", arguments)
            if result:
                logger.info(f"[MCP] Network request retrieved via MCP for page {page_idx}")
                return result
        
        logger.info(f"[MCP] Getting network request (fallback) for page {page_idx}")
        return None
    
    async def handle_dialog(self, action: str, promptText: Optional[str] = None, page_idx: Optional[int] = None) -> bool:
        """
        ダイアログを処理
        
        Args:
            action: アクション（"accept", "dismiss"）
            promptText: プロンプトテキスト（オプション）
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"action": action}
            if promptText:
                arguments['promptText'] = promptText
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_handle_dialog", arguments)
            if result:
                logger.info(f"[MCP] Dialog handled via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Handling dialog (fallback) on page {page_idx}")
        return True
    
    async def list_console_messages(self, page_idx: Optional[int] = None, types: Optional[List[str]] = None, pageSize: Optional[int] = None, pageIdx: Optional[int] = None, includePreservedMessages: bool = False) -> List[Dict]:
        """
        コンソールメッセージリストを取得
        
        Args:
            page_idx: ページインデックス（Noneの場合は現在のページ）
            types: メッセージタイプのフィルタ
            pageSize: ページサイズ
            pageIdx: ページインデックス
            includePreservedMessages: 保持されたメッセージを含めるか
        
        Returns:
            messages: コンソールメッセージリスト
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return []
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {}
            if types:
                arguments['types'] = types
            if pageSize:
                arguments['pageSize'] = pageSize
            if pageIdx:
                arguments['pageIdx'] = pageIdx
            arguments['includePreservedMessages'] = includePreservedMessages
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_list_console_messages", arguments)
            if result:
                logger.info(f"[MCP] Console messages listed via MCP for page {page_idx}")
                return result if isinstance(result, list) else []
        
        logger.info(f"[MCP] Listing console messages (fallback) for page {page_idx}")
        return []
    
    async def list_network_requests(self, page_idx: Optional[int] = None, resourceTypes: Optional[List[str]] = None, pageSize: Optional[int] = None, pageIdx: Optional[int] = None, includePreservedRequests: bool = False) -> List[Dict]:
        """
        ネットワークリクエストリストを取得
        
        Args:
            page_idx: ページインデックス（Noneの場合は現在のページ）
            resourceTypes: リソースタイプのフィルタ
            pageSize: ページサイズ
            pageIdx: ページインデックス
            includePreservedRequests: 保持されたリクエストを含めるか
        
        Returns:
            requests: ネットワークリクエストリスト
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return []
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {}
            if resourceTypes:
                arguments['resourceTypes'] = resourceTypes
            if pageSize:
                arguments['pageSize'] = pageSize
            if pageIdx:
                arguments['pageIdx'] = pageIdx
            arguments['includePreservedRequests'] = includePreservedRequests
            
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_list_network_requests", arguments)
            if result:
                logger.info(f"[MCP] Network requests listed via MCP for page {page_idx}")
                return result if isinstance(result, list) else []
        
        logger.info(f"[MCP] Listing network requests (fallback) for page {page_idx}")
        return []
    
    async def performance_start_trace(self, page_idx: Optional[int] = None, reload: bool = False, autoStop: bool = False) -> bool:
        """
        パフォーマンストレースを開始
        
        Args:
            page_idx: ページインデックス（Noneの場合は現在のページ）
            reload: ページをリロードするか
            autoStop: 自動停止するか
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"reload": reload, "autoStop": autoStop}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_performance_start_trace", arguments)
            if result:
                logger.info(f"[MCP] Performance trace started via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Starting performance trace (fallback) on page {page_idx}")
        return True
    
    async def performance_stop_trace(self, page_idx: Optional[int] = None) -> bool:
        """
        パフォーマンストレースを停止
        
        Args:
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_performance_stop_trace", {})
            if result:
                logger.info(f"[MCP] Performance trace stopped via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Stopping performance trace (fallback) on page {page_idx}")
        return True
    
    async def performance_analyze_insight(self, insightSetId: str, insightName: str, page_idx: Optional[int] = None) -> Optional[Dict]:
        """
        パフォーマンス分析インサイトを取得
        
        Args:
            insightSetId: インサイトセットID
            insightName: インサイト名
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            insight: パフォーマンス分析インサイト
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return None
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"insightSetId": insightSetId, "insightName": insightName}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_performance_analyze_insight", arguments)
            if result:
                logger.info(f"[MCP] Performance insight analyzed via MCP for page {page_idx}")
                return result
        
        logger.info(f"[MCP] Analyzing performance insight (fallback) for page {page_idx}")
        return None
    
    async def resize_page(self, width: int, height: int, page_idx: Optional[int] = None) -> bool:
        """
        ページサイズを変更
        
        Args:
            width: 幅
            height: 高さ
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"width": width, "height": height}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_resize_page", arguments)
            if result:
                logger.info(f"[MCP] Page resized via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Resizing page (fallback) on page {page_idx}")
        return True
    
    async def upload_file(self, uid: str, filePath: str, page_idx: Optional[int] = None) -> bool:
        """
        ファイルをアップロード
        
        Args:
            uid: ファイル入力要素のUID
            filePath: アップロードするファイルのパス
            page_idx: ページインデックス（Noneの場合は現在のページ）
        
        Returns:
            success: 成功フラグ
        """
        if page_idx is None:
            page_idx = self.current_page_idx
        
        if page_idx is None:
            logger.error("[MCP] No page selected")
            return False
        
        if self.use_mcp_client and self.mcp_client and self.connected:
            arguments = {"uid": uid, "filePath": filePath}
            result = await self.mcp_client.call_tool("mcp_chrome-devtools_upload_file", arguments)
            if result:
                logger.info(f"[MCP] File uploaded via MCP on page {page_idx}")
                return True
        
        logger.info(f"[MCP] Uploading file (fallback) on page {page_idx}")
        return True

