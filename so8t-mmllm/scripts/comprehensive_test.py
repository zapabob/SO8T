#!/usr/bin/env python3
"""
SO8TLLM 
llama.cpp
"""

import subprocess
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any

class SO8TComprehensiveTester:
    """SO8T"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
    
    def test_ollama_models(self) -> Dict[str, Any]:
        """Ollama"""
        print(" Ollama...")
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                models = []
                lines = result.stdout.strip().split('\n')[1:]  # 
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            models.append({
                                "name": parts[0],
                                "id": parts[1],
                                "size": parts[2],
                                "modified": " ".join(parts[3:]) if len(parts) > 3 else ""
                            })
                
                print(f" : {len(models)}")
                return {
                    "success": True,
                    "model_count": len(models),
                    "models": models,
                    "raw_output": result.stdout
                }
            else:
                print(f" : {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
        except Exception as e:
            print(f" : {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_model_inference(self, model_name: str, test_prompts: List[str]) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        results = []
        for i, prompt in enumerate(test_prompts):
            print(f"   {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["ollama", "run", model_name, prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                end_time = time.time()
                
                if result.returncode == 0:
                    print(f"    ({end_time - start_time:.2f})")
                    results.append({
                        "prompt": prompt,
                        "response": result.stdout,
                        "success": True,
                        "response_time": end_time - start_time
                    })
                else:
                    print(f"   : {result.stderr}")
                    results.append({
                        "prompt": prompt,
                        "response": result.stderr,
                        "success": False,
                        "response_time": 0
                    })
            except subprocess.TimeoutExpired:
                print(f"   ")
                results.append({
                    "prompt": prompt,
                    "response": "Timeout",
                    "success": False,
                    "response_time": 60
                })
            except Exception as e:
                print(f"   : {str(e)}")
                results.append({
                    "prompt": prompt,
                    "response": str(e),
                    "success": False,
                    "response_time": 0
                })
        
        success_count = sum(1 for r in results if r["success"])
        avg_response_time = sum(r["response_time"] for r in results if r["success"]) / max(success_count, 1)
        
        print(f" : {success_count}/{len(results)}  (: {avg_response_time:.2f})")
        
        return {
            "success": success_count > 0,
            "success_count": success_count,
            "total_count": len(results),
            "success_rate": success_count / len(results),
            "avg_response_time": avg_response_time,
            "results": results
        }
    
    def test_so8t_capabilities(self, model_name: str) -> Dict[str, Any]:
        """SO8T"""
        print(f" SO8T: {model_name}")
        
        so8t_test_prompts = [
            "SO(8)",
            "Vector RepresentationSpinor+ Representation",
            "",
            "",
            "PET"
        ]
        
        return self.test_model_inference(model_name, so8t_test_prompts)
    
    def test_multimodal_capabilities(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        multimodal_test_prompts = [
            "",
            "",
            "",
            "OCR",
            ""
        ]
        
        return self.test_model_inference(model_name, multimodal_test_prompts)
    
    def test_mathematical_reasoning(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        math_test_prompts = [
            "2x + 5 = 13 ",
            "",
            "",
            "",
            ""
        ]
        
        return self.test_model_inference(model_name, math_test_prompts)
    
    def test_ethical_reasoning(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        ethical_test_prompts = [
            "AI",
            "",
            "",
            "AI",
            ""
        ]
        
        return self.test_model_inference(model_name, ethical_test_prompts)
    
    def test_performance_metrics(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        # 
        long_prompt = "" * 10
        
        performance_tests = [
            ("", ""),
            ("", "SO(8)"),
            ("", long_prompt)
        ]
        
        results = []
        for test_name, prompt in performance_tests:
            print(f"  {test_name}...")
            result = self.test_model_inference(model_name, [prompt])
            results.append({
                "test_name": test_name,
                "prompt_length": len(prompt),
                "success": result["success"],
                "response_time": result["avg_response_time"]
            })
        
        return {
            "success": True,
            "performance_tests": results
        }
    
    def test_error_handling(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        error_test_prompts = [
            "",  # 
            "a" * 10000,  # 
            ": !@#$%^&*()_+{}|:<>?[]\\;'\",./",  # 
            "English mixed test 123",  # 
            ": "  # 
        ]
        
        return self.test_model_inference(model_name, error_test_prompts)
    
    def run_comprehensive_test(self, model_name: str = "so8t-qwen2vl-2b") -> Dict[str, Any]:
        """"""
        print(f" SO8TLLM : {model_name}")
        print("=" * 60)
        
        # 1. Ollama
        self.test_results["ollama_models"] = self.test_ollama_models()
        
        # 2. 
        basic_prompts = [
            "SO8TLLM",
            "",
            "SO(8)"
        ]
        self.test_results["basic_inference"] = self.test_model_inference(model_name, basic_prompts)
        
        # 3. SO8T
        self.test_results["so8t_capabilities"] = self.test_so8t_capabilities(model_name)
        
        # 4. 
        self.test_results["multimodal_capabilities"] = self.test_multimodal_capabilities(model_name)
        
        # 5. 
        self.test_results["mathematical_reasoning"] = self.test_mathematical_reasoning(model_name)
        
        # 6. 
        self.test_results["ethical_reasoning"] = self.test_ethical_reasoning(model_name)
        
        # 7. 
        self.test_results["performance_metrics"] = self.test_performance_metrics(model_name)
        
        # 8. 
        self.test_results["error_handling"] = self.test_error_handling(model_name)
        
        # 
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        self.test_results["summary"] = {
            "model_name": model_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time_seconds": total_time,
            "total_tests": len(self.test_results) - 1  # summary
        }
        
        print("\n" + "=" * 60)
        print(" ")
        print("=" * 60)
        print(f": {model_name}")
        print(f": {total_time:.2f}")
        print(f": {self.test_results['summary']['total_tests']}")
        
        # 
        for test_name, result in self.test_results.items():
            if test_name != "summary":
                if isinstance(result, dict) and "success" in result:
                    status = "" if result["success"] else ""
                    print(f"{status} {test_name}: {'' if result['success'] else ''}")
                else:
                    print(f" {test_name}: ")
        
        return self.test_results
    
    def save_results(self, filename: str = None):
        """"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f" : {filename}")

def main():
    """"""
    tester = SO8TComprehensiveTester()
    
    # 
    results = tester.run_comprehensive_test("so8t-qwen2vl-2b")
    
    # 
    tester.save_results()
    
    print("\n ")

if __name__ == "__main__":
    main()
