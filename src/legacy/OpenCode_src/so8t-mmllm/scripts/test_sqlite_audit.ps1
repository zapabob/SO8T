# SO8T×マルチモーダルLLM SQLite監査テスト
# WALモード + synchronous=FULL で耐久性をテスト

param(
    [string]$OutputDir = "./sqlite_test_results",
    [string]$TestDuration = "30"
)

Write-Host "🗄️ SO8T×マルチモーダルLLM SQLite監査テスト開始..." -ForegroundColor Green

# 仮想環境のアクティベート
Write-Host "🔧 仮想環境をアクティベート中..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 出力ディレクトリの作成
Write-Host "📁 出力ディレクトリを作成中..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# SQLite監査テストスクリプトの実行
Write-Host "🎯 SQLite監査テストを実行中..." -ForegroundColor Yellow

$sqliteTestScript = @"
import sys
import os
import json
import time
import threading
import random
from datetime import datetime, timedelta
from pathlib import Path

# パスを追加
sys.path.append('src')

from audit.sqlite_logger import SQLiteAuditLogger

def test_basic_operations(audit_logger):
    """基本操作をテスト"""
    print("🔧 基本操作をテスト中...")
    
    results = []
    
    # 1. 判断ログの記録
    print("  📝 判断ログを記録中...")
    for i in range(10):
        try:
            log_id = audit_logger.log_decision(
                input_text=f"テスト入力 {i+1}",
                decision=random.choice(["ALLOW", "DENY", "ESCALATE"]),
                confidence=random.uniform(0.5, 1.0),
                reasoning=f"テスト推論 {i+1}",
                meta={"test_id": i+1, "timestamp": datetime.now().isoformat()}
            )
            results.append({"operation": "log_decision", "id": log_id, "success": True})
        except Exception as e:
            results.append({"operation": "log_decision", "id": None, "success": False, "error": str(e)})
    
    # 2. ポリシー更新
    print("  📋 ポリシーを更新中...")
    for i in range(3):
        try:
            policy_id = audit_logger.update_policy(
                policy_name=f"test_policy_{i+1}",
                policy_version=f"1.{i}",
                policy_content={
                    "rule_1": f"test_value_{i+1}",
                    "rule_2": f"test_config_{i+1}",
                    "updated": True
                }
            )
            results.append({"operation": "update_policy", "id": policy_id, "success": True})
        except Exception as e:
            results.append({"operation": "update_policy", "id": None, "success": False, "error": str(e)})
    
    # 3. アイデンティティ契約更新
    print("  📄 アイデンティティ契約を更新中...")
    for i in range(2):
        try:
            contract_id = audit_logger.update_identity_contract(
                contract_name=f"test_contract_{i+1}",
                contract_version=f"2.{i}",
                contract_content={
                    "role": f"test_role_{i+1}",
                    "capabilities": ["test_capability_1", "test_capability_2"],
                    "limitations": ["test_limitation_1", "test_limitation_2"]
                }
            )
            results.append({"operation": "update_contract", "id": contract_id, "success": True})
        except Exception as e:
            results.append({"operation": "update_contract", "id": None, "success": False, "error": str(e)})
    
    # 4. 監査ログ記録
    print("  📊 監査ログを記録中...")
    for i in range(5):
        try:
            log_id = audit_logger.log_audit(
                change_type=f"test_change_{i+1}",
                change_description=f"テスト変更 {i+1}",
                change_data={
                    "test_id": i+1,
                    "change_type": f"test_change_{i+1}",
                    "timestamp": datetime.now().isoformat()
                }
            )
            results.append({"operation": "log_audit", "id": log_id, "success": True})
        except Exception as e:
            results.append({"operation": "log_audit", "id": None, "success": False, "error": str(e)})
    
    return results

def test_concurrent_operations(audit_logger, num_threads=5, operations_per_thread=20):
    """並行操作をテスト"""
    print(f"🔄 並行操作をテスト中... ({num_threads}スレッド, {operations_per_thread}操作/スレッド)")
    
    results = []
    threads = []
    
    def worker_thread(thread_id):
        thread_results = []
        for i in range(operations_per_thread):
            try:
                # ランダムな操作を選択
                operation = random.choice(["log_decision", "log_audit"])
                
                if operation == "log_decision":
                    log_id = audit_logger.log_decision(
                        input_text=f"スレッド{thread_id}入力{i+1}",
                        decision=random.choice(["ALLOW", "DENY", "ESCALATE"]),
                        confidence=random.uniform(0.5, 1.0),
                        reasoning=f"スレッド{thread_id}推論{i+1}",
                        meta={"thread_id": thread_id, "operation_id": i+1}
                    )
                    thread_results.append({
                        "thread_id": thread_id,
                        "operation": operation,
                        "id": log_id,
                        "success": True
                    })
                
                elif operation == "log_audit":
                    log_id = audit_logger.log_audit(
                        change_type=f"thread_{thread_id}_change_{i+1}",
                        change_description=f"スレッド{thread_id}変更{i+1}",
                        change_data={"thread_id": thread_id, "operation_id": i+1}
                    )
                    thread_results.append({
                        "thread_id": thread_id,
                        "operation": operation,
                        "id": log_id,
                        "success": True
                    })
                
                # 少し待機
                time.sleep(random.uniform(0.001, 0.01))
                
            except Exception as e:
                thread_results.append({
                    "thread_id": thread_id,
                    "operation": operation,
                    "id": None,
                    "success": False,
                    "error": str(e)
                })
        
        results.extend(thread_results)
    
    # スレッドを開始
    for thread_id in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(thread_id,))
        threads.append(thread)
        thread.start()
    
    # 全スレッドの完了を待機
    for thread in threads:
        thread.join()
    
    return results

def test_data_retrieval(audit_logger):
    """データ取得をテスト"""
    print("📊 データ取得をテスト中...")
    
    results = []
    
    try:
        # アクティブなポリシーを取得
        policies = audit_logger.get_active_policies()
        results.append({
            "operation": "get_active_policies",
            "count": len(policies),
            "success": True
        })
        print(f"  📋 アクティブポリシー: {len(policies)}個")
        
    except Exception as e:
        results.append({
            "operation": "get_active_policies",
            "count": 0,
            "success": False,
            "error": str(e)
        })
    
    try:
        # アクティブな契約を取得
        contracts = audit_logger.get_active_contracts()
        results.append({
            "operation": "get_active_contracts",
            "count": len(contracts),
            "success": True
        })
        print(f"  📄 アクティブ契約: {len(contracts)}個")
        
    except Exception as e:
        results.append({
            "operation": "get_active_contracts",
            "count": 0,
            "success": False,
            "error": str(e)
        })
    
    try:
        # 判断統計を取得
        stats = audit_logger.get_decision_stats(days=1)
        results.append({
            "operation": "get_decision_stats",
            "stats": stats,
            "success": True
        })
        print(f"  📈 判断統計: {stats}")
        
    except Exception as e:
        results.append({
            "operation": "get_decision_stats",
            "stats": {},
            "success": False,
            "error": str(e)
        })
    
    return results

def test_durability(audit_logger, test_duration=30):
    """耐久性をテスト"""
    print(f"⏱️ 耐久性テスト中... ({test_duration}秒)")
    
    results = []
    start_time = time.time()
    operation_count = 0
    
    while time.time() - start_time < test_duration:
        try:
            # 連続的な操作を実行
            audit_logger.log_decision(
                input_text=f"耐久性テスト入力 {operation_count}",
                decision=random.choice(["ALLOW", "DENY", "ESCALATE"]),
                confidence=random.uniform(0.5, 1.0),
                reasoning=f"耐久性テスト推論 {operation_count}",
                meta={"test_type": "durability", "operation_count": operation_count}
            )
            operation_count += 1
            
            # 定期的に統計を取得
            if operation_count % 100 == 0:
                stats = audit_logger.get_decision_stats(days=1)
                print(f"  📊 {operation_count}操作完了, 総判断数: {stats.get('total_decisions', 0)}")
            
            # 短い待機
            time.sleep(0.01)
            
        except Exception as e:
            results.append({
                "operation": "durability_test",
                "operation_count": operation_count,
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            })
            break
    
    results.append({
        "operation": "durability_test",
        "operation_count": operation_count,
        "success": True,
        "elapsed_time": time.time() - start_time
    })
    
    print(f"  ✅ 耐久性テスト完了: {operation_count}操作, {time.time() - start_time:.2f}秒")
    return results

def analyze_results(basic_results, concurrent_results, retrieval_results, durability_results):
    """結果を分析"""
    print("\\n📊 SQLite監査テスト結果分析")
    print("=" * 50)
    
    # 基本操作の分析
    basic_success = [r for r in basic_results if r.get('success', False)]
    print(f"🔧 基本操作:")
    print(f"  成功率: {len(basic_success)}/{len(basic_results)} ({len(basic_success)/len(basic_results)*100:.1f}%)")
    
    # 並行操作の分析
    concurrent_success = [r for r in concurrent_results if r.get('success', False)]
    print(f"🔄 並行操作:")
    print(f"  成功率: {len(concurrent_success)}/{len(concurrent_results)} ({len(concurrent_success)/len(concurrent_results)*100:.1f}%)")
    
    # データ取得の分析
    retrieval_success = [r for r in retrieval_results if r.get('success', False)]
    print(f"📊 データ取得:")
    print(f"  成功率: {len(retrieval_success)}/{len(retrieval_results)} ({len(retrieval_success)/len(retrieval_results)*100:.1f}%)")
    
    # 耐久性の分析
    durability_success = [r for r in durability_results if r.get('success', False)]
    if durability_success:
        total_operations = sum(r.get('operation_count', 0) for r in durability_success)
        total_time = sum(r.get('elapsed_time', 0) for r in durability_success)
        avg_ops_per_sec = total_operations / total_time if total_time > 0 else 0
        print(f"⏱️ 耐久性:")
        print(f"  総操作数: {total_operations}")
        print(f"  総時間: {total_time:.2f}秒")
        print(f"  平均操作/秒: {avg_ops_per_sec:.2f}")
    
    # 総合分析
    all_results = basic_results + concurrent_results + retrieval_results + durability_results
    all_success = [r for r in all_results if r.get('success', False)]
    overall_success_rate = len(all_success) / len(all_results) if all_results else 0.0
    
    print(f"\\n📈 総合結果:")
    print(f"  総成功率: {overall_success_rate:.3f}")
    
    return {
        "basic_success_rate": len(basic_success) / len(basic_results) if basic_results else 0.0,
        "concurrent_success_rate": len(concurrent_success) / len(concurrent_results) if concurrent_results else 0.0,
        "retrieval_success_rate": len(retrieval_success) / len(retrieval_results) if retrieval_results else 0.0,
        "durability_operations": sum(r.get('operation_count', 0) for r in durability_success),
        "overall_success_rate": overall_success_rate
    }

def main():
    print("🗄️ SO8T×マルチモーダルLLM SQLite監査テスト開始...")
    
    # 監査ロガーを初期化
    print("🔧 監査ロガーを初期化中...")
    audit_logger = SQLiteAuditLogger(
        db_path="$OutputDir/audit_test.db",
        synchronous="FULL",
        journal_mode="WAL"
    )
    
    # 各テストを実行
    print("\\n🎯 基本操作テスト開始...")
    basic_results = test_basic_operations(audit_logger)
    
    print("\\n🎯 並行操作テスト開始...")
    concurrent_results = test_concurrent_operations(audit_logger, num_threads=3, operations_per_thread=10)
    
    print("\\n🎯 データ取得テスト開始...")
    retrieval_results = test_data_retrieval(audit_logger)
    
    print("\\n🎯 耐久性テスト開始...")
    durability_results = test_durability(audit_logger, test_duration=$TestDuration)
    
    # 結果を分析
    analysis = analyze_results(basic_results, concurrent_results, retrieval_results, durability_results)
    
    # 結果を保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_duration": $TestDuration,
        "basic_results": basic_results,
        "concurrent_results": concurrent_results,
        "retrieval_results": retrieval_results,
        "durability_results": durability_results,
        "analysis": analysis
    }
    
    results_file = "$OutputDir/sqlite_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📁 結果を保存しました: {results_file}")
    print(f"📊 総合成功率: {analysis['overall_success_rate']:.3f}")
    
    print("\\n✅ SQLite監査テスト完了！")

if __name__ == "__main__":
    main()
"@

# SQLite監査テストスクリプトを実行
$sqliteTestScript | py -3

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ SQLite監査テスト完了！" -ForegroundColor Green
    Write-Host "📁 結果ディレクトリ: $OutputDir" -ForegroundColor Cyan
    Write-Host "📊 結果ファイル: $OutputDir/sqlite_test_results.json" -ForegroundColor Cyan
    Write-Host "🗄️ 監査データベース: $OutputDir/audit_test.db" -ForegroundColor Cyan
} else {
    Write-Error "❌ SQLite監査テスト中にエラーが発生しました"
    exit 1
}
