#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IntegratedReasoningPipeline 固定パーサ仕様 v1.0
See: user-provided spec (think-observation/deduction/abduction/integration + final).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Optional


STRICT_FULL_RE = re.compile(
    r'^\s*'
    r'(?:(?:<think-observation>\s*([\s\S]*?)\s*</think-observation>)\s*)?'
    r'(?:(?:<think-deduction>\s*([\s\S]*?)\s*</think-deduction>)\s*)?'
    r'(?:(?:<think-abduction>\s*([\s\S]*?)\s*</think-abduction>)\s*)?'
    r'(?:(?:<think-integration>\s*([\s\S]*?)\s*</think-integration>)\s*)?'
    r'<final>\s*([\s\S]*?)\s*</final>'
    r'\s*$'
)

FINAL_ONLY_RE = re.compile(
    r'<final>\s*([\s\S]*?)\s*</final>',
    re.IGNORECASE,
)

THINK_TAG_RE: Dict[str, re.Pattern] = {
    'observation': re.compile(r'<think-observation>\s*([\s\S]*?)\s*</think-observation>', re.IGNORECASE),
    'deduction': re.compile(r'<think-deduction>\s*([\s\S]*?)\s*</think-deduction>', re.IGNORECASE),
    'abduction': re.compile(r'<think-abduction>\s*([\s\S]*?)\s*</think-abduction>', re.IGNORECASE),
    'integration': re.compile(r'<think-integration>\s*([\s\S]*?)\s*</think-integration>', re.IGNORECASE),
}


@dataclass
class ParsedReasoning:
    final_text: str
    thinks: Dict[str, Optional[str]]
    parse_mode: str
    raw_hash: str


def _normalize_text(text: str) -> str:
    if text is None:
        return ''
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\u200b', '').replace('\ufeff', '')
    return text.strip()


def _clean_final(final_text: str) -> str:
    final_text = _normalize_text(final_text)
    for _key, pattern in THINK_TAG_RE.items():
        final_text = pattern.sub('', final_text)
    return final_text.strip()


def parse_integrated_reasoning(raw_text: str) -> ParsedReasoning:
    raw_norm = _normalize_text(raw_text)
    raw_hash = hashlib.sha256(raw_norm.encode('utf-8')).hexdigest()

    thinks: Dict[str, Optional[str]] = {
        'observation': None,
        'deduction': None,
        'abduction': None,
        'integration': None,
    }

    m_strict = STRICT_FULL_RE.match(raw_norm)
    if m_strict is not None:
        thinks['observation'] = _normalize_text(m_strict.group(1)) if m_strict.group(1) is not None else None
        thinks['deduction'] = _normalize_text(m_strict.group(2)) if m_strict.group(2) is not None else None
        thinks['abduction'] = _normalize_text(m_strict.group(3)) if m_strict.group(3) is not None else None
        thinks['integration'] = _normalize_text(m_strict.group(4)) if m_strict.group(4) is not None else None
        final_text = _clean_final(m_strict.group(5))
        return ParsedReasoning(final_text=final_text, thinks=thinks, parse_mode='strict', raw_hash=raw_hash)

    m_final = FINAL_ONLY_RE.search(raw_norm)
    if m_final is not None:
        final_text = _clean_final(m_final.group(1))
        for key, pattern in THINK_TAG_RE.items():
            all_hits = pattern.findall(raw_norm)
            if len(all_hits) > 0:
                thinks[key] = _normalize_text(all_hits[-1])
        return ParsedReasoning(final_text=final_text, thinks=thinks, parse_mode='lenient', raw_hash=raw_hash)

    return ParsedReasoning(final_text='', thinks=thinks, parse_mode='missing_final', raw_hash=raw_hash)

