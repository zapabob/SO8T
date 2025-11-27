#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodexとGemini CLIによるDeep Research /thinking形式データセット作成スクリプト

インストールされているCodexとGemini CLIにDeep Researchを実行させ、
良質な/thinking形式（思考ステップ+最終回答）のデータセットを生成
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import random

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

# ロギング設定（プロジェクトルート直下に統一）
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "create_deep_research_thinking_dataset.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Gemini APIインポート
try:
    import google.genai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("[WARNING] Gemini API not available. Install with: pip install google-genai")

# OpenAI APIインポート
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Claude APIインポート
try:
    import anthropic

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class DeepResearchThinkingGenerator:
    """Deep Researchを使用した/thinking形式データセット生成"""
    
    def __init__(
        self,
        use_codex: bool = True,
        use_gemini: bool = True,
        codex_api_type: str = "openai",
        codex_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash-exp"
    ):
        """
        Args:
            use_codex: Codex（OpenAI/Claude）を使用するか
            use_gemini: Gemini CLIを使用するか
            codex_api_type: Codex APIタイプ（"openai" or "claude"）
            codex_api_key: Codex APIキー
            gemini_api_key: Gemini APIキー
            gemini_model: Geminiモデル名
        """
        self.use_codex = use_codex
        self.use_gemini = use_gemini
        self.codex_api_type = codex_api_type
        
        # Codex API設定
        if use_codex:
            self.codex_api_key = codex_api_key or os.environ.get(f"{codex_api_type.upper()}_API_KEY")
            if not self.codex_api_key:
                logger.warning(f"[WARNING] {codex_api_type.upper()}_API_KEY not found, disabling Codex")
                self.use_codex = False
        
        # Gemini API設定
        if use_gemini:
            self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not self.gemini_api_key:
                logger.warning("[WARNING] GEMINI_API_KEY not found, disabling Gemini")
                self.use_gemini = False
            elif GEMINI_AVAILABLE:
                genai.Client.configure(api_key=self.gemini_api_key)
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                self.gemini_model = gemini_model
            else:
                logger.warning("[WARNING] google-genai not installed, disabling Gemini")
                self.use_gemini = False
        
        logger.info(f"[INIT] DeepResearchThinkingGenerator initialized")
        logger.info(f"  Codex ({codex_api_type}): {self.use_codex}")
        logger.info(f"  Gemini: {self.use_gemini}")
    
    def _call_codex_cli(self, prompt: str) -> str:
        """Codex CLIをターミナル経由で呼び出し"""
        if not self.use_codex:
            return ""
        
        try:
            # Codex CLIコマンドを試す（複数の可能性を試す）
            codex_commands = ["codex", "codex-cli", "npx codex"]
            
            for cmd_name in codex_commands:
                try:
                    # プロンプトをファイルに書き込む
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                        f.write(prompt)
                        prompt_file = f.name
                    
                    try:
                        # Codex CLIを実行（日本のドメイン別知識を優先するプロンプトを追加）
                        cmd = cmd_name.split() + [
                            "deep-research",
                            "--prompt", prompt_file,
                            "--priority", "japanese-domain-knowledge",
                            "--output-format", "thinking"
                        ]
                        
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            timeout=300  # 5分タイムアウト
                        )
                        
                        if result.returncode == 0:
                            logger.info(f"[CODEX CLI] Successfully called {cmd_name}")
                            return result.stdout.strip()
                        else:
                            logger.debug(f"[CODEX CLI] {cmd_name} failed: {result.stderr}")
                    finally:
                        if os.path.exists(prompt_file):
                            os.unlink(prompt_file)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.debug(f"[CODEX CLI] {cmd_name} error: {e}")
                    continue
            
            # CLIが見つからない場合はAPIフォールバック
            logger.info("[CODEX CLI] CLI not found, falling back to API")
            return self._call_codex_api(prompt, model=None)
        except Exception as e:
            logger.error(f"[ERROR] Codex CLI call failed: {e}")
            return ""
    
    def _call_codex_api(self, prompt: str, model: str = None) -> str:
        """Codex APIをターミナル経由で呼び出し（フォールバック）"""
        if not self.use_codex:
            return ""
        
        try:
            if self.codex_api_type == "openai":
                # OpenAI APIをcurl経由で呼び出し
                import json
                import base64
                
                # プロンプトをJSONにエンコード（日本のドメイン別知識を優先）
                messages = [
                    {"role": "system", "content": "You are an expert researcher specializing in Japanese domain knowledge. When conducting deep research, ALWAYS prioritize Japanese domain knowledge sources (kotobank, weblio, jstage, cinii, etc.) and Japanese cultural context. Provide detailed thinking steps before the final answer, with emphasis on Japanese domain-specific information."},
                    {"role": "user", "content": prompt}
                ]
                
                payload = {
                    "model": model or "gpt-4",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
                
                # curlコマンドでAPIを呼び出し
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False)
                    payload_file = f.name
                
                try:
                    # curlコマンドを試す
                    cmd = [
                        "curl",
                        "-X", "POST",
                        "https://api.openai.com/v1/chat/completions",
                        "-H", f"Authorization: Bearer {self.codex_api_key}",
                        "-H", "Content-Type: application/json",
                        "-d", f"@{payload_file}"
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        response_data = json.loads(result.stdout)
                        return response_data['choices'][0]['message']['content']
                    else:
                        # フォールバック: PowerShellのInvoke-WebRequestを使用（Windows）
                        logger.info("[INFO] curl failed, trying PowerShell Invoke-WebRequest...")
                        try:
                            with open(payload_file, 'r', encoding='utf-8') as f:
                                payload_json = f.read()
                            
                            ps_script = f'''
$headers = @{{
    "Authorization" = "Bearer {self.codex_api_key}"
    "Content-Type" = "application/json"
}}
$body = @'
{payload_json}
'@
$response = Invoke-WebRequest -Uri "https://api.openai.com/v1/chat/completions" -Method POST -Headers $headers -Body $body -UseBasicParsing
$response.Content
'''
                            
                            ps_result = subprocess.run(
                                ["powershell", "-Command", ps_script],
                                capture_output=True,
                                text=True,
                                encoding='utf-8',
                                timeout=60
                            )
                            
                            if ps_result.returncode == 0:
                                response_data = json.loads(ps_result.stdout)
                                return response_data['choices'][0]['message']['content']
                        except Exception as e2:
                            logger.warning(f"[WARNING] PowerShell fallback failed: {e2}")
                        
                        # 最終フォールバック: Pythonライブラリを使用
                        logger.info("[INFO] Falling back to Python library...")
                        if OPENAI_AVAILABLE:
                            client = openai.OpenAI(api_key=self.codex_api_key)
                            response = client.chat.completions.create(
                                model=model or "gpt-4",
                                messages=messages,
                                temperature=0.7,
                                max_tokens=2048
                            )
                            return response.choices[0].message.content
                finally:
                    if os.path.exists(payload_file):
                        os.unlink(payload_file)
                    
            elif self.codex_api_type == "claude":
                # Claude APIをcurl経由で呼び出し
                import json
                import tempfile
                
                payload = {
                    "model": model or "claude-3-opus-20240229",
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False)
                    payload_file = f.name
                
                try:
                    # curlコマンドを試す
                    cmd = [
                        "curl",
                        "-X", "POST",
                        "https://api.anthropic.com/v1/messages",
                        "-H", f"x-api-key: {self.codex_api_key}",
                        "-H", "anthropic-version: 2023-06-01",
                        "-H", "Content-Type: application/json",
                        "-d", f"@{payload_file}"
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        response_data = json.loads(result.stdout)
                        return response_data['content'][0]['text']
                    else:
                        # フォールバック: PowerShellのInvoke-WebRequestを使用（Windows）
                        logger.info("[INFO] curl failed, trying PowerShell Invoke-WebRequest...")
                        try:
                            with open(payload_file, 'r', encoding='utf-8') as f:
                                payload_json = f.read()
                            
                            ps_script = f'''
$headers = @{{
    "x-api-key" = "{self.codex_api_key}"
    "anthropic-version" = "2023-06-01"
    "Content-Type" = "application/json"
}}
$body = @'
{payload_json}
'@
$response = Invoke-WebRequest -Uri "https://api.anthropic.com/v1/messages" -Method POST -Headers $headers -Body $body -UseBasicParsing
$response.Content
'''
                            
                            ps_result = subprocess.run(
                                ["powershell", "-Command", ps_script],
                                capture_output=True,
                                text=True,
                                encoding='utf-8',
                                timeout=60
                            )
                            
                            if ps_result.returncode == 0:
                                response_data = json.loads(ps_result.stdout)
                                return response_data['content'][0]['text']
                        except Exception as e2:
                            logger.warning(f"[WARNING] PowerShell fallback failed: {e2}")
                        
                        # 最終フォールバック: Pythonライブラリを使用
                        logger.info("[INFO] Falling back to Python library...")
                        if CLAUDE_AVAILABLE:
                            client = anthropic.Anthropic(api_key=self.codex_api_key)
                            response = client.messages.create(
                                model=model or "claude-3-opus-20240229",
                                max_tokens=2048,
                                messages=[{"role": "user", "content": prompt}]
                            )
                            return response.content[0].text
                finally:
                    if os.path.exists(payload_file):
                        os.unlink(payload_file)
                    
        except Exception as e:
            logger.error(f"[ERROR] Codex API call failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return ""
    
    async def _call_gemini_cli(self, prompt: str) -> str:
        """Gemini CLIをターミナル経由で呼び出し"""
        if not self.use_gemini:
            return ""
        
        try:
            # Gemini CLIコマンドを試す（複数の可能性を試す）
            # 一般的なCLIツール名を試す
            gemini_commands = [
                ["gemini"],
                ["gemini-cli"],
                ["google-gemini"],
                ["npx", "gemini"],
                ["npx", "@google/gemini"],
                ["node", "gemini"],
                ["python", "-m", "gemini"],
            ]
            
            for cmd_parts in gemini_commands:
                try:
                    # プロンプトをファイルに書き込む
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                        f.write(prompt)
                        prompt_file = f.name
                    
                    try:
                        # Gemini CLIを実行（Deep Researchモード、日本のドメイン別知識を優先）
                        # 一般的なCLIインターフェースを試す
                        cmd_variants = [
                            cmd_parts + ["deep-research", "--prompt", prompt_file, "--priority", "japanese-domain-knowledge", "--model", self.gemini_model],
                            cmd_parts + ["research", "--query", prompt_file, "--japanese-priority", "--model", self.gemini_model],
                            cmd_parts + ["--prompt", prompt_file, "--japanese-domain", "--model", self.gemini_model],
                            cmd_parts + [prompt_file],  # シンプルにプロンプトファイルを渡す
                        ]
                        
                        for cmd in cmd_variants:
                            try:
                                result = subprocess.run(
                                    cmd,
                                    capture_output=True,
                                    text=True,
                                    encoding='utf-8',
                                    timeout=300,  # 5分タイムアウト
                                    env=dict(os.environ, PYTHONUNBUFFERED='1')
                                )
                                
                                if result.returncode == 0 and result.stdout.strip():
                                    logger.info(f"[GEMINI CLI] Successfully called {' '.join(cmd_parts)}")
                                    return result.stdout.strip()
                                elif result.returncode == 0:
                                    logger.debug(f"[GEMINI CLI] {' '.join(cmd_parts)} returned empty output")
                            except subprocess.TimeoutExpired:
                                logger.debug(f"[GEMINI CLI] {' '.join(cmd_parts)} timed out")
                                continue
                            except Exception as e:
                                logger.debug(f"[GEMINI CLI] {' '.join(cmd_parts)} error: {e}")
                                continue
                    finally:
                        if os.path.exists(prompt_file):
                            os.unlink(prompt_file)
                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    continue
                except Exception as e:
                    logger.debug(f"[GEMINI CLI] {' '.join(cmd_parts)} check error: {e}")
                    continue
            
            # CLIが見つからない場合は空文字列を返す（APIフォールバックは呼び出し側で処理）
            logger.info("[GEMINI CLI] CLI not found, will fall back to API")
            return ""
        except Exception as e:
            logger.error(f"[ERROR] Gemini CLI call failed: {e}")
            return ""
    
    async def _call_gemini_deep_research_with_prompt(self, prompt: str) -> str:
        """Gemini APIをターミナル経由で呼び出し（フォールバック）"""
        if not self.use_gemini:
            return ""
        
        try:
            # Gemini APIをcurl経由で呼び出し
            import json
            import tempfile
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 4096
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
                payload_file = f.name
            
            try:
                # Gemini APIをcurl経由で呼び出し
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
                
                cmd = [
                    "curl",
                    "-X", "POST",
                    url,
                    "-H", "Content-Type: application/json",
                    "-d", f"@{payload_file}"
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=120
                )
                
                if result.returncode == 0:
                    response_data = json.loads(result.stdout)
                    if 'candidates' in response_data and len(response_data['candidates']) > 0:
                        return response_data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        logger.warning("[WARNING] No candidates in Gemini response")
                else:
                    # フォールバック: PowerShellのInvoke-WebRequestを使用（Windows）
                    logger.info("[INFO] curl failed, trying PowerShell Invoke-WebRequest...")
                    try:
                        with open(payload_file, 'r', encoding='utf-8') as f:
                            payload_json = f.read()
                        
                        ps_script = f'''
$body = @'
{payload_json}
'@
$response = Invoke-WebRequest -Uri "{url}" -Method POST -Headers @{{"Content-Type"="application/json"}} -Body $body -UseBasicParsing
$response.Content
'''
                        
                        ps_result = subprocess.run(
                            ["powershell", "-Command", ps_script],
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            timeout=120
                        )
                        
                        if ps_result.returncode == 0:
                            response_data = json.loads(ps_result.stdout)
                            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                                return response_data['candidates'][0]['content']['parts'][0]['text']
                    except Exception as e2:
                        logger.warning(f"[WARNING] PowerShell fallback failed: {e2}")
                
                # 最終フォールバック: Pythonライブラリを使用
                logger.info("[INFO] Falling back to Python library...")
                if GEMINI_AVAILABLE and hasattr(self, 'gemini_client'):
                    model = self.gemini_client.models.get(self.gemini_model)
                    response = model.generate_content(prompt)
                    return response.text
            finally:
                if os.path.exists(payload_file):
                    os.unlink(payload_file)
                
        except Exception as e:
            logger.error(f"[ERROR] Gemini Deep Research failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # フォールバック: Pythonライブラリを使用
            if GEMINI_AVAILABLE and hasattr(self, 'gemini_client'):
                try:
                    model = self.gemini_client.models.get(self.gemini_model)
                    response = model.generate_content(prompt)
                    return response.text
                except:
                    pass
        
        return ""
    
    def _load_japanese_domain_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """日本のドメイン別知識を読み込み"""
        domain_knowledge = []
        
        # ドメイン知識ファイルのパス
        domain_knowledge_paths = [
            Path("D:/webdataset/domain_knowledge_collected"),
            Path("D:/webdataset/processed/domain_knowledge"),
            PROJECT_ROOT / "data" / "domain_knowledge"
        ]
        
        for base_path in domain_knowledge_paths:
            if base_path.exists():
                # JSONLファイルを検索
                for jsonl_file in base_path.glob("*.jsonl"):
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    sample = json.loads(line)
                                    # クエリに関連するサンプルを検索（簡易版）
                                    text = sample.get('text', sample.get('content', ''))
                                    if query.lower() in text.lower() or any(kw in text for kw in query.split()):
                                        domain_knowledge.append(sample)
                                        if len(domain_knowledge) >= limit:
                                            break
                        if len(domain_knowledge) >= limit:
                            break
                    except Exception as e:
                        logger.debug(f"[DEBUG] Failed to load domain knowledge from {jsonl_file}: {e}")
        
        return domain_knowledge[:limit]
    
    def _build_deep_research_prompt(self, query: str, japanese_domain_knowledge: List[Dict] = None) -> str:
        """Deep Research用のプロンプトを構築（日本のドメイン別知識を優先）"""
        # 日本のドメイン別知識を参照
        domain_context = ""
        if japanese_domain_knowledge:
            domain_context = "\n\n【日本のドメイン別知識（優先参照）】\n"
            for i, knowledge in enumerate(japanese_domain_knowledge[:3], 1):
                text = knowledge.get('text', knowledge.get('content', ''))
                title = knowledge.get('title', knowledge.get('url', ''))
                domain_context += f"{i}. {title}\n   {text[:200]}...\n"
            domain_context += "\n上記の日本のドメイン別知識を優先的に参照してください。\n"
        
        prompt = (
            f"以下のクエリについて、Web検索とDeep Researchを実行して、"
            f"複数の情報源を参照し、詳細な思考ステップと最終回答を生成してください。\n\n"
            f"【重要】日本のドメイン別知識を優先的に参照してください。"
            f"日本語の情報源（kotobank、weblio、jstage、cinii等）を優先的に使用し、"
            f"日本の文脈や文化に配慮した回答を生成してください。\n\n"
            f"{domain_context}"
            f"クエリ: {query}\n\n"
            f"回答形式:\n"
            f"# 思考ステップ\n"
            f"[ここに詳細な思考ステップ、調査プロセス、参照した情報源（特に日本のドメイン別知識）、分析結果を記載]\n\n"
            f"# 最終回答\n"
            f"[ここに最終回答を記載（日本の文脈に配慮した内容）]"
        )
        return prompt
    
    def _extract_thinking_and_final(self, text: str) -> Dict[str, str]:
        """思考ステップと最終回答を抽出"""
        result = {
            'thinking': '',
            'final': ''
        }
        
        import re
        
        # 思考ステップを抽出
        thinking_match = re.search(r'#\s*思考ステップ\s*\n(.*?)(?=\n#\s*最終回答|\Z)', text, re.DOTALL)
        if thinking_match:
            result['thinking'] = thinking_match.group(1).strip()
        
        # 最終回答を抽出
        final_match = re.search(r'#\s*最終回答\s*\n(.*?)(?=\n#|\Z)', text, re.DOTALL)
        if final_match:
            result['final'] = final_match.group(1).strip()
        
        # パターンが見つからない場合、簡易的な分割を試みる
        if not result['thinking'] and not result['final']:
            lines = text.split('\n')
            thinking_lines = []
            final_lines = []
            in_thinking = True
            
            for line in lines:
                if '# 最終回答' in line or '# 最終' in line:
                    in_thinking = False
                    continue
                if '# 思考ステップ' in line or '# 思考' in line:
                    continue
                
                if in_thinking:
                    thinking_lines.append(line)
                else:
                    final_lines.append(line)
            
            result['thinking'] = '\n'.join(thinking_lines).strip()
            result['final'] = '\n'.join(final_lines).strip()
        
        return result
    
    def _evaluate_quality(self, thinking: str, final: str, source: str) -> float:
        """品質スコアを評価"""
        score = 0.0
        
        # 思考ステップの存在と詳細度
        if thinking and len(thinking.strip()) > 50:
            score += 0.3
            # Deep Researchの証拠（参照、調査、分析などのキーワード）
            research_keywords = ['参照', '調査', '分析', '検索', '情報源', '資料', '文献', '研究', '確認']
            if any(kw in thinking for kw in research_keywords):
                score += 0.2
        
        # 最終回答の存在と適切な長さ
        if final and len(final.strip()) > 10:
            score += 0.3
        
        # 思考ステップの詳細度（長さ）
        if thinking:
            thinking_length = len(thinking)
            if 100 <= thinking_length <= 2000:
                score += 0.1
            elif thinking_length > 2000:
                score += 0.05
        
        # 最終回答の適切な長さ
        if final:
            final_length = len(final)
            if 20 <= final_length <= 1000:
                score += 0.1
        
        return min(score, 1.0)
    
    async def generate_thinking_sample(self, query: str) -> Optional[Dict]:
        """/thinking形式のサンプルを生成（CodexとGeminiの両方を使用、日本のドメイン別知識を優先）"""
        results = []
        
        # 日本のドメイン別知識を読み込み
        japanese_domain_knowledge = self._load_japanese_domain_knowledge(query, limit=5)
        if japanese_domain_knowledge:
            logger.info(f"[DOMAIN] Loaded {len(japanese_domain_knowledge)} Japanese domain knowledge samples")
        
        # Codexで生成（CLIを優先、フォールバックでAPI）
        if self.use_codex:
            logger.info(f"[CODEX] Generating sample for query: {query[:50]}...")
            codex_prompt = self._build_deep_research_prompt(query, japanese_domain_knowledge)
            # CLIを優先的に試す
            codex_response = self._call_codex_cli(codex_prompt)
            if not codex_response:
                # CLIが失敗した場合はAPIフォールバック
                codex_response = self._call_codex_api(codex_prompt)
            
            if codex_response:
                codex_parts = self._extract_thinking_and_final(codex_response)
                codex_quality = self._evaluate_quality(
                    codex_parts.get('thinking', ''),
                    codex_parts.get('final', ''),
                    'codex'
                )
                
                if codex_quality >= 0.5:
                    results.append({
                        "instruction": query,
                        "input": "",
                        "output": f"# 思考ステップ\n{codex_parts.get('thinking', '')}\n\n# 最終回答\n{codex_parts.get('final', '')}",
                        "thinking": codex_parts.get('thinking', ''),
                        "final": codex_parts.get('final', ''),
                        "quality_score": codex_quality,
                        "source": f"codex_{self.codex_api_type}",
                        "created_at": datetime.now().isoformat()
                    })
                    logger.info(f"[CODEX] Generated sample (quality: {codex_quality:.2f})")
        
        # Geminiで生成（CLIを優先、フォールバックでAPI）
        if self.use_gemini:
            logger.info(f"[GEMINI] Generating sample for query: {query[:50]}...")
            gemini_prompt = self._build_deep_research_prompt(query, japanese_domain_knowledge)
            # CLIを優先的に試す
            gemini_response = await self._call_gemini_cli(gemini_prompt)
            if not gemini_response:
                # CLIが失敗した場合はAPIフォールバック
                gemini_response = await self._call_gemini_deep_research_with_prompt(gemini_prompt)
            
            if gemini_response:
                gemini_parts = self._extract_thinking_and_final(gemini_response)
                gemini_quality = self._evaluate_quality(
                    gemini_parts.get('thinking', ''),
                    gemini_parts.get('final', ''),
                    'gemini'
                )
                
                if gemini_quality >= 0.5:
                    results.append({
                        "instruction": query,
                        "input": "",
                        "output": f"# 思考ステップ\n{gemini_parts.get('thinking', '')}\n\n# 最終回答\n{gemini_parts.get('final', '')}",
                        "thinking": gemini_parts.get('thinking', ''),
                        "final": gemini_parts.get('final', ''),
                        "quality_score": gemini_quality,
                        "source": "gemini_deep_research",
                        "created_at": datetime.now().isoformat()
                    })
                    logger.info(f"[GEMINI] Generated sample (quality: {gemini_quality:.2f})")
        
        # 最高品質のサンプルを返す
        if results:
            best_sample = max(results, key=lambda x: x.get('quality_score', 0.0))
            return best_sample
        
        return None
    
    async def generate_thinking_samples(
        self,
        queries: List[str],
        num_samples_per_query: int = 1
    ) -> List[Dict]:
        """
        /thinking形式のサンプルを生成
        
        Args:
            queries: クエリリスト
            num_samples_per_query: クエリあたりのサンプル数
        
        Returns:
            /thinking形式サンプルのリスト
        """
        samples = []
        
        for i, query in enumerate(queries):
            logger.info(f"[GENERATE] Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            for sample_idx in range(num_samples_per_query):
                sample = await self.generate_thinking_sample(query)
                
                if sample:
                    samples.append(sample)
                    logger.info(f"[OK] Generated sample {sample_idx+1} for query {i+1} (quality: {sample.get('quality_score', 0.0):.2f}, source: {sample.get('source', 'unknown')})")
                
                # APIレート制限対策
                await asyncio.sleep(1)
        
        return samples


async def main_async():
    parser = argparse.ArgumentParser(
        description="Generate /thinking format dataset using Codex and Gemini CLI Deep Research"
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        required=True,
        help="Path to queries file (JSONL format, one query per line, or JSON with 'instruction' field)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output file path (JSONL format)"
    )
    parser.add_argument(
        "--use-codex",
        action="store_true",
        default=True,
        help="Use Codex (OpenAI/Claude API)"
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        default=True,
        help="Use Gemini CLI Deep Research"
    )
    parser.add_argument(
        "--codex-api-type",
        type=str,
        choices=["openai", "claude"],
        default="openai",
        help="Codex API type (default: openai)"
    )
    parser.add_argument(
        "--codex-api-key",
        type=str,
        default=None,
        help="Codex API key (if not provided, uses environment variable)"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (if not provided, uses environment variable)"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Gemini model name (default: gemini-2.0-flash-exp)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per query (default: 1)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.6,
        help="Minimum quality score threshold (default: 0.6)"
    )
    
    args = parser.parse_args()
    
    # クエリを読み込み
    queries = []
    with open(args.queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    # JSONL形式を試す
                    data = json.loads(line)
                    if isinstance(data, dict):
                        query = data.get('instruction', data.get('prompt', data.get('query', data.get('text', ''))))
                    else:
                        query = str(data)
                except json.JSONDecodeError:
                    # プレーンテキストとして扱う
                    query = line.strip()
                
                if query:
                    queries.append(query)
    
    logger.info(f"[MAIN] Loaded {len(queries)} queries from {args.queries_file}")
    
    if not queries:
        logger.error("[ERROR] No queries found in input file")
        return
    
    # データセット生成器を初期化
    generator = DeepResearchThinkingGenerator(
        use_codex=args.use_codex,
        use_gemini=args.use_gemini,
        codex_api_type=args.codex_api_type,
        codex_api_key=args.codex_api_key,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model
    )
    
    # /thinking形式サンプルを生成
    logger.info(f"[MAIN] Generating {args.num_samples} samples per query...")
    samples = await generator.generate_thinking_samples(
        queries=queries,
        num_samples_per_query=args.num_samples
    )
    
    logger.info(f"[MAIN] Generated {len(samples)} samples before filtering")
    
    if not samples:
        error_msg = "No samples generated. Check API keys and network connectivity."
        logger.error(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)
    
    # 品質フィルタリング
    if args.min_quality > 0:
        filtered_samples = [s for s in samples if s.get('quality_score', 0.0) >= args.min_quality]
        logger.info(f"[FILTER] Filtered {len(samples)} -> {len(filtered_samples)} samples (min_quality={args.min_quality})")
        samples = filtered_samples
    
    if not samples:
        error_msg = f"No samples passed quality filter (min_quality={args.min_quality})"
        logger.error(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)
    
    # 出力ファイルに保存
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[SAVE] Writing {len(samples)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # ファイルサイズを確認
    file_size = output_file.stat().st_size
    logger.info(f"[SUCCESS] Generated {len(samples)} /thinking format samples")
    logger.info(f"[SAVE] Saved to {output_file} ({file_size:,} bytes)")
    
    if file_size == 0:
        error_msg = f"Output file is empty after writing {len(samples)} samples"
        logger.error(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)
    
    # 統計情報を出力
    if samples:
        avg_quality = sum(s.get('quality_score', 0.0) for s in samples) / len(samples)
        logger.info(f"[STATS] Average quality score: {avg_quality:.2f}")
        
        source_counts = {}
        for sample in samples:
            source = sample.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info(f"[STATS] Source distribution: {source_counts}")
        
        quality_ranges = {
            "0.0-0.5": 0,
            "0.5-0.7": 0,
            "0.7-0.9": 0,
            "0.9-1.0": 0
        }
        for sample in samples:
            q = sample.get('quality_score', 0.0)
            if q < 0.5:
                quality_ranges["0.0-0.5"] += 1
            elif q < 0.7:
                quality_ranges["0.5-0.7"] += 1
            elif q < 0.9:
                quality_ranges["0.7-0.9"] += 1
            else:
                quality_ranges["0.9-1.0"] += 1
        
        logger.info(f"[STATS] Quality distribution:")
        for range_name, count in quality_ranges.items():
            logger.info(f"  {range_name}: {count} samples")


def main():
    """メイン関数（非同期ラッパー）"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

