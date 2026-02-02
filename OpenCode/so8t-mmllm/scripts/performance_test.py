#!/usr/bin/env python3
"""
SO8TLLM 

"""

import subprocess
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any

class SO8TPerformanceTester:
    """SO8T"""
    
    def __init__(self):
        self.results = {}
    
    def test_response_time(self, model_name: str, prompts: List[str], iterations: int = 3) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        response_times = []
        for i, prompt in enumerate(prompts):
            print(f"   {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            prompt_times = []
            for iteration in range(iterations):
                try:
                    start_time = time.time()
                    result = subprocess.run(
                        ["ollama", "run", model_name, prompt],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    end_time = time.time()
                    
                    if result.returncode == 0:
                        response_time = end_time - start_time
                        prompt_times.append(response_time)
                        print(f"     {iteration+1}: {response_time:.2f}")
                    else:
                        print(f"     {iteration+1}: ")
                        
                except subprocess.TimeoutExpired:
                    print(f"     {iteration+1}: ")
                except Exception as e:
                    print(f"     {iteration+1}:  - {str(e)}")
            
            if prompt_times:
                avg_time = statistics.mean(prompt_times)
                min_time = min(prompt_times)
                max_time = max(prompt_times)
                std_dev = statistics.stdev(prompt_times) if len(prompt_times) > 1 else 0
                
                response_times.append({
                    "prompt": prompt,
                    "iterations": len(prompt_times),
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "std_dev": std_dev,
                    "times": prompt_times
                })
                
                print(f"    : {avg_time:.2f} (: {min_time:.2f}, : {max_time:.2f}, : {std_dev:.2f})")
        
        # 
        all_times = [rt["avg_time"] for rt in response_times]
        overall_avg = statistics.mean(all_times) if all_times else 0
        overall_min = min(all_times) if all_times else 0
        overall_max = max(all_times) if all_times else 0
        overall_std = statistics.stdev(all_times) if len(all_times) > 1 else 0
        
        print(f" :  {overall_avg:.2f} (: {overall_min:.2f}, : {overall_max:.2f})")
        
        return {
            "success": len(response_times) > 0,
            "total_prompts": len(prompts),
            "successful_prompts": len(response_times),
            "overall_avg": overall_avg,
            "overall_min": overall_min,
            "overall_max": overall_max,
            "overall_std": overall_std,
            "prompt_results": response_times
        }
    
    def test_memory_usage(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        try:
            # 
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 
                for line in lines:
                    if model_name in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            size_str = parts[2]
                            print(f"  : {size_str}")
                            return {
                                "success": True,
                                "model_size": size_str,
                                "raw_output": line
                            }
            
            print("  ")
            return {"success": False, "error": "Model info not found"}
            
        except Exception as e:
            print(f"  : {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_concurrent_requests(self, model_name: str, prompt: str, num_requests: int = 3) -> Dict[str, Any]:
        """"""
        print(f" : {model_name} ({num_requests})")
        
        import threading
        import queue
        
        results = queue.Queue()
        
        def run_request(request_id: int):
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["ollama", "run", model_name, prompt],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                end_time = time.time()
                
                results.put({
                    "request_id": request_id,
                    "success": result.returncode == 0,
                    "response_time": end_time - start_time,
                    "response": result.stdout if result.returncode == 0 else result.stderr
                })
            except Exception as e:
                results.put({
                    "request_id": request_id,
                    "success": False,
                    "response_time": 0,
                    "response": str(e)
                })
        
        # 
        threads = []
        start_time = time.time()
        
        for i in range(num_requests):
            thread = threading.Thread(target=run_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 
        request_results = []
        while not results.empty():
            request_results.append(results.get())
        
        success_count = sum(1 for r in request_results if r["success"])
        response_times = [r["response_time"] for r in request_results if r["success"]]
        
        print(f"  : {success_count}/{num_requests}")
        print(f"  : {total_time:.2f}")
        if response_times:
            print(f"  : {statistics.mean(response_times):.2f}")
        
        return {
            "success": success_count > 0,
            "total_requests": num_requests,
            "successful_requests": success_count,
            "total_time": total_time,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "request_results": request_results
        }
    
    def test_long_context(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        # 
        long_prompt = "" * 100
        
        try:
            start_time = time.time()
            result = subprocess.run(
                ["ollama", "run", model_name, long_prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            success = result.returncode == 0
            
            print(f"  : {len(long_prompt)}")
            print(f"  : {response_time:.2f}")
            print(f"  : {success}")
            
            return {
                "success": success,
                "prompt_length": len(long_prompt),
                "response_time": response_time,
                "response_length": len(result.stdout) if success else 0,
                "error": result.stderr if not success else None
            }
            
        except subprocess.TimeoutExpired:
            print("  ")
            return {
                "success": False,
                "prompt_length": len(long_prompt),
                "response_time": 60,
                "timeout": True
            }
        except Exception as e:
            print(f"  : {str(e)}")
            return {
                "success": False,
                "prompt_length": len(long_prompt),
                "response_time": 0,
                "error": str(e)
            }
    
    def run_performance_test(self, model_name: str = "so8t-qwen2vl-2b") -> Dict[str, Any]:
        """"""
        print(f" SO8TLLM : {model_name}")
        print("=" * 60)
        
        # 
        test_prompts = [
            "",
            "SO(8)",
            ": 2x + 5 = 13",
            "AI",
            ""
        ]
        
        # 1. 
        self.results["response_time"] = self.test_response_time(model_name, test_prompts, iterations=3)
        
        # 2. 
        self.results["memory_usage"] = self.test_memory_usage(model_name)
        
        # 3. 
        self.results["concurrent_requests"] = self.test_concurrent_requests(
            model_name, 
            "SO8T", 
            num_requests=3
        )
        
        # 4. 
        self.results["long_context"] = self.test_long_context(model_name)
        
        # 
        end_time = datetime.now()
        
        self.results["summary"] = {
            "model_name": model_name,
            "test_time": end_time.isoformat(),
            "total_tests": 4
        }
        
        print("\n" + "=" * 60)
        print(" ")
        print("=" * 60)
        print(f": {model_name}")
        print(f": {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 
        for test_name, result in self.results.items():
            if test_name != "summary":
                if isinstance(result, dict) and "success" in result:
                    status = "" if result["success"] else ""
                    print(f"{status} {test_name}: {'' if result['success'] else ''}")
                else:
                    print(f" {test_name}: ")
        
        return self.results
    
    def save_results(self, filename: str = None):
        """"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f" : {filename}")

def main():
    """"""
    tester = SO8TPerformanceTester()
    
    # 
    results = tester.run_performance_test("so8t-qwen2vl-2b")
    
    # 
    tester.save_results()
    
    print("\n ")

if __name__ == "__main__":
    main()
