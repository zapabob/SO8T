#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chromeブラウザヘッダー生成ユーティリティ

WebスクレイピングをChromeブラウザに偽装するためのヘッダーを生成
"""

from typing import Dict, Optional


def get_chrome_headers(referer: Optional[str] = None) -> Dict[str, str]:
    """
    Chromeブラウザのヘッダーを生成
    
    Args:
        referer: リファラーURL（オプション）
    
    Returns:
        headers: Chromeのヘッダー辞書
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0'
    }
    
    if referer:
        headers['Referer'] = referer
    
    return headers


def get_chrome_api_headers() -> Dict[str, str]:
    """
    ChromeブラウザのAPIリクエスト用ヘッダーを生成
    
    Returns:
        headers: ChromeのAPIリクエスト用ヘッダー辞書
    """
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Connection': 'keep-alive'
    }


def get_chrome_user_agent() -> str:
    """
    ChromeブラウザのUser-Agent文字列を取得
    
    Returns:
        user_agent: ChromeのUser-Agent文字列
    """
    return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'







