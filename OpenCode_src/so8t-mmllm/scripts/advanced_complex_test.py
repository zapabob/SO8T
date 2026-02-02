#!/usr/bin/env python3
"""
SO8TLLM 

"""

import subprocess
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any

class SO8TAdvancedComplexTester:
    """SO8T"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
    
    def test_complex_mathematical_reasoning(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        complex_math_prompts = [
            " dy/dx = 2xy ",
            " z = 3 + 4i ",
            " A = [[1,2],[3,4]] ",
            " N(,) ",
            "^ e^(-x) dx ",
            " Z ",
            ""
        ]
        
        return self._test_model_inference(model_name, complex_math_prompts, "")
    
    def test_advanced_logical_reasoning(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        logical_prompts = [
            "",
            "((P  Q)  (Q  R))  (P  R) ",
            "x(P(x)  Q(x))  x(P(x)  Q(x)) ",
            "",
            "",
            "2",
            "nn"
        ]
        
        return self._test_model_inference(model_name, logical_prompts, "")
    
    def test_philosophical_reasoning(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        philosophical_prompts = [
            "AI",
            "",
            "",
            "",
            "",
            "",
            ""
        ]
        
        return self._test_model_inference(model_name, philosophical_prompts, "")
    
    def test_scientific_reasoning(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        scientific_prompts = [
            "GPS",
            "",
            "",
            "IPCC",
            "DNA",
            "",
            ""
        ]
        
        return self._test_model_inference(model_name, scientific_prompts, "")
    
    def test_creative_problem_solving(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        creative_prompts = [
            "2050",
            "",
            "",
            "",
            "",
            "",
            "AI"
        ]
        
        return self._test_model_inference(model_name, creative_prompts, "")
    
    def test_ethical_dilemma_analysis(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        ethical_prompts = [
            "",
            "AI",
            "",
            "",
            "AIAIAI",
            "AIAI",
            "AI"
        ]
        
        return self._test_model_inference(model_name, ethical_prompts, "")
    
    def test_cross_domain_integration(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        cross_domain_prompts = [
            "",
            "",
            "",
            "",
            "",
            "",
            "AI"
        ]
        
        return self._test_model_inference(model_name, cross_domain_prompts, "")
    
    def test_edge_cases_and_boundaries(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        edge_case_prompts = [
            "",
            "0",
            "",
            "",
            "",
            "",
            "AI"
        ]
        
        return self._test_model_inference(model_name, edge_case_prompts, "")
    
    def test_meta_cognitive_reasoning(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        meta_cognitive_prompts = [
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        ]
        
        return self._test_model_inference(model_name, meta_cognitive_prompts, "")
    
    def test_stress_and_complexity(self, model_name: str) -> Dict[str, Any]:
        """"""
        print(f" : {model_name}")
        
        # 
        stress_prompts = [
            "" + "" * 20,
            "" + "" * 15,
            "" + "" * 10,
            "" + "" * 12,
            "" + "" * 8
        ]
        
        return self._test_model_inference(model_name, stress_prompts, "")
    
    def _test_model_inference(self, model_name: str, prompts: List[str], test_category: str) -> Dict[str, Any]:
        """"""
        results = []
        total_time = 0
        
        for i, prompt in enumerate(prompts):
            print(f"  {test_category} {i+1}/{len(prompts)}: {prompt[:80]}...")
            
            try:
                start_time = time.time()
                result = subprocess.run(
                    ["ollama", "run", model_name, prompt],
                    capture_output=True,
                    text=True,
                    timeout=120  # 
                )
                end_time = time.time()
                response_time = end_time - start_time
                total_time += response_time
                
                if result.returncode == 0:
                    print(f"      ({response_time:.2f})")
                    results.append({
                        "prompt": prompt,
                        "response": result.stdout,
                        "success": True,
                        "response_time": response_time,
                        "response_length": len(result.stdout)
                    })
                else:
                    print(f"     : {result.stderr}")
                    results.append({
                        "prompt": prompt,
                        "response": result.stderr,
                        "success": False,
                        "response_time": response_time,
                        "response_length": 0
                    })
            except subprocess.TimeoutExpired:
                print(f"     ")
                results.append({
                    "prompt": prompt,
                    "response": "Timeout",
                    "success": False,
                    "response_time": 120,
                    "response_length": 0
                })
            except Exception as e:
                print(f"     : {str(e)}")
                results.append({
                    "prompt": prompt,
                    "response": str(e),
                    "success": False,
                    "response_time": 0,
                    "response_length": 0
                })
        
        success_count = sum(1 for r in results if r["success"])
        avg_response_time = total_time / len(prompts)
        avg_response_length = sum(r["response_length"] for r in results if r["success"]) / max(success_count, 1)
        
        print(f"   {test_category}: {success_count}/{len(prompts)}  (: {avg_response_time:.2f}, : {avg_response_length:.0f})")
        
        return {
            "success": success_count > 0,
            "success_count": success_count,
            "total_count": len(prompts),
            "success_rate": success_count / len(prompts),
            "avg_response_time": avg_response_time,
            "avg_response_length": avg_response_length,
            "total_time": total_time,
            "results": results
        }
    
    def run_advanced_complex_test(self, model_name: str = "so8t-qwen2vl-2b") -> Dict[str, Any]:
        """"""
        print(f" SO8TLLM : {model_name}")
        print("=" * 80)
        
        # 1. 
        self.test_results["complex_mathematical"] = self.test_complex_mathematical_reasoning(model_name)
        
        # 2. 
        self.test_results["advanced_logical"] = self.test_advanced_logical_reasoning(model_name)
        
        # 3. 
        self.test_results["philosophical"] = self.test_philosophical_reasoning(model_name)
        
        # 4. 
        self.test_results["scientific"] = self.test_scientific_reasoning(model_name)
        
        # 5. 
        self.test_results["creative_problem_solving"] = self.test_creative_problem_solving(model_name)
        
        # 6. 
        self.test_results["ethical_dilemma"] = self.test_ethical_dilemma_analysis(model_name)
        
        # 7. 
        self.test_results["cross_domain"] = self.test_cross_domain_integration(model_name)
        
        # 8. 
        self.test_results["edge_cases"] = self.test_edge_cases_and_boundaries(model_name)
        
        # 9. 
        self.test_results["meta_cognitive"] = self.test_meta_cognitive_reasoning(model_name)
        
        # 10. 
        self.test_results["stress_complexity"] = self.test_stress_and_complexity(model_name)
        
        # 
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        # 
        total_tests = sum(r["total_count"] for r in self.test_results.values() if isinstance(r, dict) and "total_count" in r)
        total_success = sum(r["success_count"] for r in self.test_results.values() if isinstance(r, dict) and "success_count" in r)
        overall_success_rate = total_success / total_tests if total_tests > 0 else 0
        
        self.test_results["summary"] = {
            "model_name": model_name,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time_seconds": total_time,
            "total_tests": total_tests,
            "total_success": total_success,
            "overall_success_rate": overall_success_rate,
            "test_categories": len(self.test_results) - 1
        }
        
        print("\n" + "=" * 80)
        print(" ")
        print("=" * 80)
        print(f": {model_name}")
        print(f": {total_time:.2f}")
        print(f": {total_tests}")
        print(f": {total_success}")
        print(f": {overall_success_rate:.2%}")
        print(f": {len(self.test_results) - 1}")
        
        # 
        for test_name, result in self.test_results.items():
            if test_name != "summary" and isinstance(result, dict) and "success_rate" in result:
                status = "" if result["success"] else ""
                print(f"{status} {test_name}: {result['success_rate']:.2%} ({result['success_count']}/{result['total_count']})")
        
        return self.test_results
    
    def save_results(self, filename: str = None):
        """"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_complex_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f" : {filename}")

def main():
    """"""
    tester = SO8TAdvancedComplexTester()
    
    # 
    results = tester.run_advanced_complex_test("so8t-qwen2vl-2b")
    
    # 
    tester.save_results()
    
    print("\n ")

if __name__ == "__main__":
    main()
