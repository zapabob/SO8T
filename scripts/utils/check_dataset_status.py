#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""データセット状況確認スクリプト"""

from pathlib import Path
import json
from collections import Counter

def main():
    fp = Path('D:/webdataset/processed/four_class/four_class_improved_20251109_101005.jsonl')
    
    if not fp.exists():
        print(f"ファイルが見つかりません: {fp}")
        return
    
    lines = fp.read_text(encoding='utf-8').strip().split('\n')
    samples = [json.loads(l) for l in lines if l.strip()]
    
    print("="*80)
    print("ESCALATION分類の詳細")
    print("="*80)
    
    escalation_samples = [s for s in samples if s.get('quadruple_classification', {}).get('four_class_label') == 'ESCALATION']
    print(f"ESCALATIONサンプル数: {len(escalation_samples)}")
    
    if escalation_samples:
        reasons = {}
        domains = {}
        categories = {}
        text_lengths = []
        
        for s in escalation_samples:
            reason = s.get('quadruple_classification', {}).get('escalation_reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
            
            domain = s.get('domain', 'unknown')
            domains[domain] = domains.get(domain, 0) + 1
            
            category = s.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            text_length = len(s.get('text', ''))
            text_lengths.append(text_length)
        
        print(f"\nESCALATION理由:")
        for k, v in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k}: {v}")
        
        print(f"\nESCALATIONドメイン（上位10）:")
        for k, v in sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {k}: {v}")
        
        print(f"\nESCALATIONカテゴリ（上位10）:")
        for k, v in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {k}: {v}")
        
        print(f"\nESCALATIONテキスト長統計:")
        print(f"  最小: {min(text_lengths)}文字")
        print(f"  最大: {max(text_lengths)}文字")
        print(f"  平均: {sum(text_lengths)/len(text_lengths):.1f}文字")
        print(f"  中央値: {sorted(text_lengths)[len(text_lengths)//2]}文字")
        print(f"  1000文字超: {sum(1 for l in text_lengths if l > 1000)} ({sum(1 for l in text_lengths if l > 1000)/len(text_lengths)*100:.1f}%)")

if __name__ == "__main__":
    main()







































