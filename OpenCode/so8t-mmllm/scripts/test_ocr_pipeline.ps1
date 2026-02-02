# SO8T×マルチモーダルLLM OCRパイプラインテスト
# OpenCV + Tesseract でローカル画像処理をテスト

param(
    [string]$TestImageDir = "./test_images",
    [string]$OutputDir = "./ocr_test_results",
    [string]$Languages = "jpn+eng"
)

Write-Host "🔍 SO8T×マルチモーダルLLM OCRパイプラインテスト開始..." -ForegroundColor Green

# 仮想環境のアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# OCRパイプラインテストスクリプトの実行
Write-Host "🎯 OCRパイプラインテストを実行中..." -ForegroundColor Yellow

$ocrTestScript = @"
import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2

# パスを追加
sys.path.append('src')

from io.ocr_summary import OCRSummaryProcessor
from audit.sqlite_logger import SQLiteAuditLogger

def create_test_images(output_dir):
    """テスト用画像を作成"""
    print("🖼️ テスト用画像を作成中...")
    
    test_images = []
    
    # 日本語テキスト画像
    img1 = Image.new('RGB', (400, 200), color='white')
    draw1 = ImageDraw.Draw(img1)
    try:
        # 日本語フォントを試行
        font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    draw1.text((20, 50), "これは日本語のテストテキストです", fill='black', font=font)
    draw1.text((20, 100), "OCR処理のテストを行います", fill='black', font=font)
    
    img1_path = os.path.join(output_dir, "test_japanese.jpg")
    img1.save(img1_path)
    test_images.append({
        "path": img1_path,
        "type": "japanese",
        "expected_text": "これは日本語のテストテキストです OCR処理のテストを行います"
    })
    
    # 英語テキスト画像
    img2 = Image.new('RGB', (400, 200), color='white')
    draw2 = ImageDraw.Draw(img2)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw2.text((20, 50), "This is English test text", fill='black', font=font)
    draw2.text((20, 100), "Testing OCR processing pipeline", fill='black', font=font)
    
    img2_path = os.path.join(output_dir, "test_english.jpg")
    img2.save(img2_path)
    test_images.append({
        "path": img2_path,
        "type": "english",
        "expected_text": "This is English test text Testing OCR processing pipeline"
    })
    
    # 混合言語画像
    img3 = Image.new('RGB', (400, 200), color='white')
    draw3 = ImageDraw.Draw(img3)
    try:
        jp_font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 20)
        en_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except:
        jp_font = en_font = ImageFont.load_default()
    
    draw3.text((20, 50), "Mixed 日本語 English Text", fill='black', font=en_font)
    draw3.text((20, 100), "OCR テスト Test", fill='black', font=en_font)
    
    img3_path = os.path.join(output_dir, "test_mixed.jpg")
    img3.save(img3_path)
    test_images.append({
        "path": img3_path,
        "type": "mixed",
        "expected_text": "Mixed 日本語 English Text OCR テスト Test"
    })
    
    # ノイズ画像（低品質）
    img4 = Image.new('RGB', (400, 200), color='white')
    draw4 = ImageDraw.Draw(img4)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw4.text((20, 50), "Noisy Image Test", fill='black', font=font)
    draw4.text((20, 100), "Low Quality Text", fill='black', font=font)
    
    # ノイズを追加
    img4_array = np.array(img4)
    noise = np.random.randint(0, 50, img4_array.shape, dtype=np.uint8)
    img4_array = np.clip(img4_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img4 = Image.fromarray(img4_array)
    
    img4_path = os.path.join(output_dir, "test_noisy.jpg")
    img4.save(img4_path)
    test_images.append({
        "path": img4_path,
        "type": "noisy",
        "expected_text": "Noisy Image Test Low Quality Text"
    })
    
    print(f"✅ {len(test_images)}個のテスト画像を作成しました")
    return test_images

def test_ocr_processor(ocr_processor, test_images):
    """OCRプロセッサをテスト"""
    print("🔍 OCRプロセッサをテスト中...")
    
    results = []
    
    for i, img_info in enumerate(test_images):
        print(f"  📷 画像 {i+1}: {img_info['type']}")
        
        try:
            # OCR処理を実行
            summary = ocr_processor.process_image(img_info['path'])
            
            # 結果を評価
            result = {
                "image_id": i,
                "image_path": img_info['path'],
                "image_type": img_info['type'],
                "expected_text": img_info['expected_text'],
                "ocr_result": summary,
                "success": True
            }
            
            # 簡易評価
            detected_text = summary.get('text', '')
            confidence = summary.get('confidence', 0.0)
            
            # テキストの長さ評価
            text_length_score = min(len(detected_text) / 50.0, 1.0)
            
            # 信頼度評価
            confidence_score = min(confidence / 100.0, 1.0)
            
            # 期待テキストとの類似度（簡易版）
            expected_words = set(img_info['expected_text'].lower().split())
            detected_words = set(detected_text.lower().split())
            similarity = len(expected_words & detected_words) / max(len(expected_words), 1)
            
            # 総合スコア
            overall_score = (text_length_score * 0.3 + confidence_score * 0.4 + similarity * 0.3)
            
            result.update({
                "text_length_score": text_length_score,
                "confidence_score": confidence_score,
                "similarity_score": similarity,
                "overall_score": overall_score,
                "detected_text": detected_text,
                "confidence": confidence
            })
            
            print(f"    信頼度: {confidence:.1f}%")
            print(f"    検出テキスト: {detected_text[:50]}...")
            print(f"    スコア: {overall_score:.3f}")
            
        except Exception as e:
            print(f"    ❌ エラー: {str(e)}")
            result = {
                "image_id": i,
                "image_path": img_info['path'],
                "image_type": img_info['type'],
                "expected_text": img_info['expected_text'],
                "success": False,
                "error": str(e),
                "overall_score": 0.0
            }
        
        results.append(result)
    
    return results

def test_ocr_with_audit(ocr_processor, audit_logger, test_images):
    """監査ログ付きOCRテスト"""
    print("🗄️ 監査ログ付きOCRテスト中...")
    
    results = []
    
    for i, img_info in enumerate(test_images):
        print(f"  📷 監査付き画像 {i+1}: {img_info['type']}")
        
        try:
            # OCR処理を実行
            summary = ocr_processor.process_image(img_info['path'])
            
            # 監査ログに記録
            audit_logger.log_decision(
                input_text=f"OCR processing: {img_info['path']}",
                decision="ALLOW",
                confidence=summary.get('confidence', 0.0) / 100.0,
                reasoning=f"OCR processing completed for {img_info['type']} image",
                meta={
                    "image_type": img_info['type'],
                    "ocr_confidence": summary.get('confidence', 0.0),
                    "text_length": len(summary.get('text', '')),
                    "language": summary.get('lang', 'unknown')
                }
            )
            
            result = {
                "image_id": i,
                "image_path": img_info['path'],
                "image_type": img_info['type'],
                "ocr_result": summary,
                "audit_logged": True,
                "success": True
            }
            
            print(f"    ✅ OCR処理完了、監査ログ記録済み")
            
        except Exception as e:
            print(f"    ❌ エラー: {str(e)}")
            result = {
                "image_id": i,
                "image_path": img_info['path'],
                "image_type": img_info['type'],
                "success": False,
                "error": str(e),
                "audit_logged": False
            }
        
        results.append(result)
    
    return results

def analyze_results(ocr_results, audit_results):
    """結果を分析"""
    print("\\n📊 OCRパイプライン結果分析")
    print("=" * 50)
    
    # OCR結果の分析
    successful_ocr = [r for r in ocr_results if r.get('success', False)]
    if successful_ocr:
        scores = [r.get('overall_score', 0.0) for r in successful_ocr]
        confidences = [r.get('confidence', 0.0) for r in successful_ocr]
        
        print(f"📈 OCR処理統計:")
        print(f"  成功率: {len(successful_ocr)}/{len(ocr_results)} ({len(successful_ocr)/len(ocr_results)*100:.1f}%)")
        print(f"  平均スコア: {np.mean(scores):.3f}")
        print(f"  平均信頼度: {np.mean(confidences):.1f}%")
        print(f"  最高スコア: {np.max(scores):.3f}")
        print(f"  最低スコア: {np.min(scores):.3f}")
        
        # 画像タイプ別分析
        type_scores = {}
        for result in successful_ocr:
            img_type = result.get('image_type', 'unknown')
            if img_type not in type_scores:
                type_scores[img_type] = []
            type_scores[img_type].append(result.get('overall_score', 0.0))
        
        print(f"\\n📊 画像タイプ別スコア:")
        for img_type, scores in type_scores.items():
            print(f"  {img_type}: {np.mean(scores):.3f} (n={len(scores)})")
    
    # 監査ログ結果の分析
    successful_audit = [r for r in audit_results if r.get('success', False)]
    print(f"\\n🗄️ 監査ログ統計:")
    print(f"  成功率: {len(successful_audit)}/{len(audit_results)} ({len(successful_audit)/len(audit_results)*100:.1f}%)")
    
    return {
        "ocr_success_rate": len(successful_ocr) / len(ocr_results) if ocr_results else 0.0,
        "audit_success_rate": len(successful_audit) / len(audit_results) if audit_results else 0.0,
        "overall_success_rate": (len(successful_ocr) + len(successful_audit)) / (len(ocr_results) + len(audit_results)) if (ocr_results and audit_results) else 0.0
    }

def main():
    print("🔍 SO8T×マルチモーダルLLM OCRパイプラインテスト開始...")
    
    # テスト画像を作成
    test_images = create_test_images('$OutputDir')
    
    # OCRプロセッサを初期化
    print("🔧 OCRプロセッサを初期化中...")
    ocr_processor = OCRSummaryProcessor(
        tesseract_config="--oem 3 --psm 6",
        languages="$Languages",
        min_confidence=30.0
    )
    
    # 監査ロガーを初期化
    print("🗄️ 監査ロガーを初期化中...")
    audit_logger = SQLiteAuditLogger(
        db_path="$OutputDir/ocr_audit.db",
        synchronous="FULL",
        journal_mode="WAL"
    )
    
    # OCRプロセッサをテスト
    print("\\n🎯 OCRプロセッサテスト開始...")
    ocr_results = test_ocr_processor(ocr_processor, test_images)
    
    # 監査ログ付きOCRテスト
    print("\\n🎯 監査ログ付きOCRテスト開始...")
    audit_results = test_ocr_with_audit(ocr_processor, audit_logger, test_images)
    
    # 結果を分析
    analysis = analyze_results(ocr_results, audit_results)
    
    # 結果を保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_images": test_images,
        "ocr_results": ocr_results,
        "audit_results": audit_results,
        "analysis": analysis
    }
    
    results_file = "$OutputDir/ocr_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📁 結果を保存しました: {results_file}")
    print(f"📊 総合成功率: {analysis['overall_success_rate']:.3f}")
    
    print("\\n✅ OCRパイプラインテスト完了！")

if __name__ == "__main__":
    main()
"@

# OCRパイプラインテストスクリプトを実行
$ocrTestScript | py -3

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ OCRパイプラインテスト完了！" -ForegroundColor Green
    Write-Host "📁 結果ディレクトリ: $OutputDir" -ForegroundColor Cyan
    Write-Host "📊 結果ファイル: $OutputDir/ocr_test_results.json" -ForegroundColor Cyan
    Write-Host "🗄️ 監査データベース: $OutputDir/ocr_audit.db" -ForegroundColor Cyan
} else {
    Write-Error "❌ OCRパイプラインテスト中にエラーが発生しました"
    exit 1
}
