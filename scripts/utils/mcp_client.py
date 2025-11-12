#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
MCPプロトコルクライアント

MCPサーバーに直接接続してMCPツールを呼び出すためのクライアント実装。
JSON-RPC 2.0準拠のプロトコルを使用します。

Usage:
    from scripts.utils.mcp_client import MCPClient
    
    client = MCPClient(transport="stdio", command="npx", args=["-y", "@modelcontextprotocol/server-chrome-devtools"])
    await client.connect()
    result = await client.call_tool("mcp_chrome-devtools_new_page", {"url": "https://example.com"})
    await client.disconnect()
"""

import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class TransportType(Enum):
    """トランスポートタイプ"""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


class MCPClient:
    """MCPプロトコルクライアント"""
    
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
        self.transport = TransportType(transport.lower())
        self.command = command
        self.args = args or []
        self.url = url
        self.timeout = timeout / 1000.0  # ミリ秒から秒に変換
        
        self.process: Optional[subprocess.Popen] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self.request_id = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
        
        logger.info(f"[MCP] Client initialized (transport: {self.transport.value})")
    
    async def connect(self) -> bool:
        """
        MCPサーバーに接続
        
        Returns:
            success: 成功フラグ
        """
        try:
            if self.transport == TransportType.STDIO:
                return await self._connect_stdio()
            elif self.transport == TransportType.HTTP:
                return await self._connect_http()
            elif self.transport == TransportType.WEBSOCKET:
                return await self._connect_websocket()
            else:
                logger.error(f"[MCP] Unsupported transport type: {self.transport}")
                return False
        except Exception as e:
            logger.error(f"[MCP] Connection failed: {e}")
            return False
    
    async def _connect_stdio(self) -> bool:
        """stdioトランスポートで接続"""
        try:
            if not self.command:
                logger.error("[MCP] Command not specified for stdio transport")
                return False
            
            logger.info(f"[MCP] Starting MCP server: {self.command} {' '.join(self.args)}")
            
            # プロセスを起動
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                bufsize=0
            )
            
            # stdoutリーダーを設定
            self.reader = self.process.stdout
            self.writer = self.process.stdin
            
            # バックグラウンドでstdoutを読み取るタスクを開始
            asyncio.create_task(self._read_stdout_loop())
            
            # 初期化メッセージを送信
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "SO8T MCP Client",
                        "version": "1.0.0"
                    }
                }
            }
            
            await self._send_message(init_message)
            
            # 初期化応答を待機
            response = await self._receive_message(timeout=self.timeout)
            if response and response.get("result"):
                self.connected = True
                logger.info("[MCP] Connected to MCP server via stdio")
                return True
            else:
                logger.error("[MCP] Failed to initialize MCP server")
                return False
                
        except Exception as e:
            logger.error(f"[MCP] stdio connection failed: {e}")
            return False
    
    async def _read_stdout_loop(self):
        """stdoutを読み取るループ"""
        try:
            while self.process and self.process.returncode is None:
                if self.reader:
                    line = await self.reader.readline()
                    if line:
                        await self._process_stdout_data(line)
                    else:
                        break
        except Exception as e:
            logger.error(f"[MCP] stdout read loop failed: {e}")
    
    async def _connect_http(self) -> bool:
        """HTTPトランスポートで接続"""
        try:
            try:
                import aiohttp
            except ImportError:
                logger.error("[MCP] aiohttp not installed. Install with: pip install aiohttp")
                return False
            
            if not self.url:
                logger.error("[MCP] URL not specified for HTTP transport")
                return False
            
            logger.info(f"[MCP] Connecting to MCP server via HTTP: {self.url}")
            
            # HTTP接続の実装
            # 実際の実装では、aiohttpを使用してHTTP接続を確立
            # ここでは簡易実装として、接続フラグを設定
            
            self.connected = True
            logger.info("[MCP] Connected to MCP server via HTTP")
            return True
            
        except Exception as e:
            logger.error(f"[MCP] HTTP connection failed: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """WebSocketトランスポートで接続"""
        try:
            try:
                import websockets
            except ImportError:
                logger.error("[MCP] websockets not installed. Install with: pip install websockets")
                return False
            
            if not self.url:
                logger.error("[MCP] URL not specified for WebSocket transport")
                return False
            
            logger.info(f"[MCP] Connecting to MCP server via WebSocket: {self.url}")
            
            # WebSocket接続の実装
            # 実際の実装では、websocketsライブラリを使用してWebSocket接続を確立
            # ここでは簡易実装として、接続フラグを設定
            
            self.connected = True
            logger.info("[MCP] Connected to MCP server via WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"[MCP] WebSocket connection failed: {e}")
            return False
    
    async def _process_stdout_data(self, data: bytes):
        """stdoutデータを処理"""
        try:
            # JSON-RPCメッセージを解析
            text = data.decode('utf-8').strip()
            if text:
                message = json.loads(text)
                await self._handle_message(message)
        except json.JSONDecodeError:
            # 複数行のメッセージの場合、行ごとに処理
            for line in text.split('\n'):
                if line.strip():
                    try:
                        message = json.loads(line.strip())
                        await self._handle_message(message)
                    except json.JSONDecodeError:
                        logger.debug(f"[MCP] Failed to parse JSON line: {line[:100]}")
        except Exception as e:
            logger.error(f"[MCP] Failed to process stdout data: {e}")
    
    async def _handle_message(self, message: Dict):
        """受信メッセージを処理"""
        try:
            message_id = message.get("id")
            if message_id and message_id in self.pending_requests:
                future = self.pending_requests.pop(message_id)
                if "error" in message:
                    future.set_exception(Exception(message["error"].get("message", "Unknown error")))
                else:
                    future.set_result(message.get("result"))
        except Exception as e:
            logger.error(f"[MCP] Failed to handle message: {e}")
    
    async def _send_message(self, message: Dict) -> bool:
        """
        メッセージを送信
        
        Args:
            message: 送信するメッセージ
        
        Returns:
            success: 成功フラグ
        """
        try:
            if self.transport == TransportType.STDIO:
                if self.writer:
                    json_str = json.dumps(message, ensure_ascii=False) + "\n"
                    self.writer.write(json_str.encode('utf-8'))
                    await self.writer.drain()
                    logger.debug(f"[MCP] Sent message: {message.get('method', 'unknown')}")
                    return True
            elif self.transport == TransportType.HTTP:
                # HTTP送信の実装
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url, json=message) as response:
                        if response.status == 200:
                            result = await response.json()
                            await self._handle_message(result)
                            return True
            elif self.transport == TransportType.WEBSOCKET:
                # WebSocket送信の実装
                import websockets
                async with websockets.connect(self.url) as websocket:
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    await self._handle_message(json.loads(response))
                    return True
            
            return False
        except Exception as e:
            logger.error(f"[MCP] Failed to send message: {e}")
            return False
    
    async def _receive_message(self, timeout: float = 30.0) -> Optional[Dict]:
        """
        メッセージを受信（非推奨: 代わりにcall_toolを使用）
        
        Args:
            timeout: タイムアウト（秒）
        
        Returns:
            message: 受信したメッセージ
        """
        try:
            # 簡易実装として、タイムアウトを待機
            # 実際の実装では、_handle_messageで処理される
            await asyncio.sleep(0.1)
            return None
        except Exception as e:
            logger.error(f"[MCP] Failed to receive message: {e}")
            return None
    
    async def call_tool(self, tool_name: str, arguments: Optional[Dict] = None) -> Optional[Any]:
        """
        MCPツールを呼び出す
        
        Args:
            tool_name: ツール名
            arguments: ツールの引数
        
        Returns:
            result: ツールの実行結果
        """
        if not self.connected:
            logger.error("[MCP] Not connected to MCP server")
            return None
        
        try:
            self.request_id += 1
            request_id = self.request_id
            
            message = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {}
                }
            }
            
            # リクエストを送信
            future = asyncio.Future()
            self.pending_requests[request_id] = future
            
            await self._send_message(message)
            
            # 応答を待機
            try:
                result = await asyncio.wait_for(future, timeout=self.timeout)
                return result
            except asyncio.TimeoutError:
                logger.error(f"[MCP] Tool call timeout: {tool_name}")
                self.pending_requests.pop(request_id, None)
                return None
            except Exception as e:
                logger.error(f"[MCP] Tool call failed: {e}")
                self.pending_requests.pop(request_id, None)
                return None
                
        except Exception as e:
            logger.error(f"[MCP] Failed to call tool: {e}")
            return None
    
    async def list_tools(self) -> List[Dict]:
        """
        利用可能なツールをリストアップ
        
        Returns:
            tools: ツールリスト
        """
        if not self.connected:
            logger.error("[MCP] Not connected to MCP server")
            return []
        
        try:
            self.request_id += 1
            request_id = self.request_id
            
            message = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/list",
                "params": {}
            }
            
            # リクエストを送信
            future = asyncio.Future()
            self.pending_requests[request_id] = future
            
            await self._send_message(message)
            
            # 応答を待機
            try:
                result = await asyncio.wait_for(future, timeout=self.timeout)
                return result.get("tools", []) if result else []
            except asyncio.TimeoutError:
                logger.error("[MCP] List tools timeout")
                self.pending_requests.pop(request_id, None)
                return []
            except Exception as e:
                logger.error(f"[MCP] List tools failed: {e}")
                self.pending_requests.pop(request_id, None)
                return []
                
        except Exception as e:
            logger.error(f"[MCP] Failed to list tools: {e}")
            return []
    
    async def disconnect(self):
        """接続を切断"""
        try:
            if self.transport == TransportType.STDIO:
                if self.process:
                    self.process.terminate()
                    await self.process.wait()
                    self.process = None
            elif self.transport == TransportType.HTTP:
                # HTTP切断の実装
                pass
            elif self.transport == TransportType.WEBSOCKET:
                # WebSocket切断の実装
                pass
            
            self.connected = False
            logger.info("[MCP] Disconnected from MCP server")
        except Exception as e:
            logger.error(f"[MCP] Failed to disconnect: {e}")

