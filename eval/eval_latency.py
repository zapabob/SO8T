"""
SO8T Latency Evaluation Script

This script evaluates the latency and performance characteristics of SO8T models
across different hardware configurations and model variants.

Key features:
- Comprehensive latency evaluation across multiple model variants
- Hardware-specific performance analysis (RTX3060, CPU, Edge devices)
- Throughput and memory usage measurements
- Integration with llama.cpp for GGUF model evaluation
- Detailed performance reporting and visualization
"""

import os
import json
import logging
import argparse
import subprocess
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import GPUtil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyEvaluator:
    """
    Latency evaluator for SO8T models.
    
    Evaluates performance characteristics across different model variants
    and hardware configurations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the latency evaluator.
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test data
        self.test_data = []
        
        # Results storage
        self.results = {}
        
        # Hardware information
        self.hardware_info = self._get_hardware_info()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config["output_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_dir / "latency_evaluation.log")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information for the current system."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "platform": os.name,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
        }
        
        # GPU information
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "gpu_memory": [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
            })
        else:
            info["cuda_available"] = False
        
        # Try to get GPU info using GPUtil
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info["gpu_utilization"] = [gpu.load for gpu in gpus]
                info["gpu_memory_used"] = [gpu.memoryUsed for gpu in gpus]
                info["gpu_memory_total"] = [gpu.memoryTotal for gpu in gpus]
        except:
            pass
        
        return info
    
    def _load_test_data(self):
        """Load test data for latency evaluation."""
        test_data_path = self.config["test_data_path"]
        
        logger.info(f"Loading test data from {test_data_path}")
        
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        self.test_data.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Loaded {len(self.test_data)} test samples")
    
    def _evaluate_pytorch_model(
        self,
        model_path: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a PyTorch model variant.
        
        Args:
            model_path: Path to the model
            model_name: Name of the model variant
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating PyTorch model: {model_name}")
        
        # Load model
        try:
            from models.so8t_model import load_so8t_model
            model = load_so8t_model(model_path)
            model.eval()
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {}
        
        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["base_model_name"],
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            return {}
        
        # Evaluation results
        processing_times = []
        memory_usage = []
        gpu_memory_usage = []
        
        # Warmup
        logger.info("Warming up model...")
        for _ in range(5):
            try:
                sample = self.test_data[0]
                inputs = tokenizer(
                    f"Context: {sample['context']}\nUser Request: {sample['user_request']}",
                    max_length=self.config.get("max_length", 512),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    _ = model(**inputs)
            except Exception as e:
                logger.warning(f"Warmup error: {e}")
                break
        
        # Evaluate on test data
        for sample in tqdm(self.test_data, desc=f"Evaluating {model_name}"):
            try:
                # Prepare input
                inputs = tokenizer(
                    f"Context: {sample['context']}\nUser Request: {sample['user_request']}",
                    max_length=self.config.get("max_length", 512),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Measure memory before
                memory_before = psutil.virtual_memory().used
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated()
                
                # Measure processing time
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Measure memory after
                memory_after = psutil.virtual_memory().used
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated()
                
                # Store results
                processing_times.append(processing_time)
                memory_usage.append(memory_after - memory_before)
                if torch.cuda.is_available():
                    gpu_memory_usage.append(gpu_memory_after - gpu_memory_before)
                
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        # Calculate metrics
        metrics = self._calculate_pytorch_metrics(
            processing_times,
            memory_usage,
            gpu_memory_usage,
            model_name
        )
        
        return metrics
    
    def _evaluate_gguf_model(
        self,
        model_path: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a GGUF model variant using llama.cpp.
        
        Args:
            model_path: Path to the GGUF model
            model_name: Name of the model variant
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating GGUF model: {model_name}")
        
        # Check if llama.cpp is available
        llama_cpp_path = self.config.get("llama_cpp_path", "llama-cli")
        
        try:
            # Test llama.cpp availability
            result = subprocess.run([llama_cpp_path, "--help"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("llama.cpp not available or not working")
                return {}
        except Exception as e:
            logger.error(f"Error testing llama.cpp: {e}")
            return {}
        
        # Evaluation results
        processing_times = []
        memory_usage = []
        
        # Prepare test prompts
        test_prompts = []
        for sample in self.test_data:
            prompt = f"Context: {sample['context']}\nUser Request: {sample['user_request']}\nResponse:"
            test_prompts.append(prompt)
        
        # Evaluate each prompt
        for prompt in tqdm(test_prompts, desc=f"Evaluating {model_name}"):
            try:
                # Measure memory before
                memory_before = psutil.virtual_memory().used
                
                # Measure processing time
                start_time = time.time()
                
                # Run llama.cpp
                cmd = [
                    llama_cpp_path,
                    "-m", model_path,
                    "-p", prompt,
                    "-n", "50",  # Generate 50 tokens
                    "--temp", "0.7",
                    "--top-p", "0.9"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Measure memory after
                memory_after = psutil.virtual_memory().used
                
                # Store results
                processing_times.append(processing_time)
                memory_usage.append(memory_after - memory_before)
                
            except Exception as e:
                logger.warning(f"Error processing prompt: {e}")
                continue
        
        # Calculate metrics
        metrics = self._calculate_gguf_metrics(
            processing_times,
            memory_usage,
            model_name
        )
        
        return metrics
    
    def _calculate_pytorch_metrics(
        self,
        processing_times: List[float],
        memory_usage: List[int],
        gpu_memory_usage: List[int],
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate metrics for PyTorch model evaluation."""
        if not processing_times:
            return {}
        
        processing_times = np.array(processing_times)
        memory_usage = np.array(memory_usage)
        gpu_memory_usage = np.array(gpu_memory_usage) if gpu_memory_usage else np.array([0])
        
        metrics = {
            "model_name": model_name,
            "model_type": "pytorch",
            "num_samples": len(processing_times),
            "average_processing_time": np.mean(processing_times),
            "median_processing_time": np.median(processing_times),
            "std_processing_time": np.std(processing_times),
            "min_processing_time": np.min(processing_times),
            "max_processing_time": np.max(processing_times),
            "throughput": 1.0 / np.mean(processing_times),
            "average_memory_usage": np.mean(memory_usage),
            "max_memory_usage": np.max(memory_usage),
            "average_gpu_memory_usage": np.mean(gpu_memory_usage) if len(gpu_memory_usage) > 0 else 0,
            "max_gpu_memory_usage": np.max(gpu_memory_usage) if len(gpu_memory_usage) > 0 else 0
        }
        
        # Calculate percentiles
        for p in [50, 90, 95, 99]:
            metrics[f"p{p}_processing_time"] = np.percentile(processing_times, p)
        
        return metrics
    
    def _calculate_gguf_metrics(
        self,
        processing_times: List[float],
        memory_usage: List[int],
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate metrics for GGUF model evaluation."""
        if not processing_times:
            return {}
        
        processing_times = np.array(processing_times)
        memory_usage = np.array(memory_usage)
        
        metrics = {
            "model_name": model_name,
            "model_type": "gguf",
            "num_samples": len(processing_times),
            "average_processing_time": np.mean(processing_times),
            "median_processing_time": np.median(processing_times),
            "std_processing_time": np.std(processing_times),
            "min_processing_time": np.min(processing_times),
            "max_processing_time": np.max(processing_times),
            "throughput": 1.0 / np.mean(processing_times),
            "average_memory_usage": np.mean(memory_usage),
            "max_memory_usage": np.max(memory_usage)
        }
        
        # Calculate percentiles
        for p in [50, 90, 95, 99]:
            metrics[f"p{p}_processing_time"] = np.percentile(processing_times, p)
        
        return metrics
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.results:
            return {}
        
        # Create performance comparison table
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                "Model": model_name,
                "Type": metrics.get("model_type", "unknown"),
                "Avg Processing Time (s)": metrics.get("average_processing_time", 0),
                "Median Processing Time (s)": metrics.get("median_processing_time", 0),
                "P95 Processing Time (s)": metrics.get("p95_processing_time", 0),
                "Throughput (req/s)": metrics.get("throughput", 0),
                "Avg Memory Usage (MB)": metrics.get("average_memory_usage", 0) / (1024 * 1024),
                "Max Memory Usage (MB)": metrics.get("max_memory_usage", 0) / (1024 * 1024)
            })
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Generate report
        report = {
            "hardware_info": self.hardware_info,
            "comparison_table": df.to_dict("records"),
            "best_throughput": df.loc[df["Throughput (req/s)"].idxmax(), "Model"],
            "fastest_processing": df.loc[df["Avg Processing Time (s)"].idxmin(), "Model"],
            "most_memory_efficient": df.loc[df["Avg Memory Usage (MB)"].idxmin(), "Model"],
            "summary": {
                "total_models_evaluated": len(self.results),
                "average_processing_time": df["Avg Processing Time (s)"].mean(),
                "average_throughput": df["Throughput (req/s)"].mean(),
                "average_memory_usage": df["Avg Memory Usage (MB)"].mean()
            }
        }
        
        return report
    
    def _generate_visualizations(self):
        """Generate performance visualization plots."""
        if len(self.results) < 2:
            logger.warning("Not enough results for visualizations")
            return
        
        # Create output directory for plots
        plots_dir = Path(self.config["output_dir"]) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        
        # Create processing time comparison
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Processing time comparison
        plt.subplot(2, 2, 1)
        processing_times = [self.results[model]["average_processing_time"] for model in model_names]
        plt.bar(model_names, processing_times)
        plt.title("Average Processing Time Comparison")
        plt.ylabel("Processing Time (seconds)")
        plt.xticks(rotation=45)
        
        # Subplot 2: Throughput comparison
        plt.subplot(2, 2, 2)
        throughputs = [self.results[model]["throughput"] for model in model_names]
        plt.bar(model_names, throughputs)
        plt.title("Throughput Comparison")
        plt.ylabel("Throughput (requests/second)")
        plt.xticks(rotation=45)
        
        # Subplot 3: Memory usage comparison
        plt.subplot(2, 2, 3)
        memory_usage = [self.results[model]["average_memory_usage"] / (1024 * 1024) for model in model_names]
        plt.bar(model_names, memory_usage)
        plt.title("Average Memory Usage Comparison")
        plt.ylabel("Memory Usage (MB)")
        plt.xticks(rotation=45)
        
        # Subplot 4: P95 processing time comparison
        plt.subplot(2, 2, 4)
        p95_times = [self.results[model]["p95_processing_time"] for model in model_names]
        plt.bar(model_names, p95_times)
        plt.title("P95 Processing Time Comparison")
        plt.ylabel("P95 Processing Time (seconds)")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed performance analysis
        self._create_detailed_analysis(plots_dir)
        
        logger.info(f"Visualizations saved to {plots_dir}")
    
    def _create_detailed_analysis(self, plots_dir: Path):
        """Create detailed performance analysis plots."""
        # Create a comprehensive performance dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("SO8T Performance Analysis Dashboard", fontsize=16)
        
        model_names = list(self.results.keys())
        
        # Processing time distribution
        axes[0, 0].boxplot([self.results[model]["average_processing_time"] for model in model_names])
        axes[0, 0].set_title("Processing Time Distribution")
        axes[0, 0].set_ylabel("Processing Time (seconds)")
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        
        # Throughput vs Memory usage
        throughputs = [self.results[model]["throughput"] for model in model_names]
        memory_usage = [self.results[model]["average_memory_usage"] / (1024 * 1024) for model in model_names]
        axes[0, 1].scatter(memory_usage, throughputs)
        axes[0, 1].set_title("Throughput vs Memory Usage")
        axes[0, 1].set_xlabel("Memory Usage (MB)")
        axes[0, 1].set_ylabel("Throughput (req/s)")
        for i, model in enumerate(model_names):
            axes[0, 1].annotate(model, (memory_usage[i], throughputs[i]))
        
        # Processing time percentiles
        percentiles = [50, 90, 95, 99]
        for model in model_names:
            p_values = [self.results[model][f"p{p}_processing_time"] for p in percentiles]
            axes[0, 2].plot(percentiles, p_values, marker='o', label=model)
        axes[0, 2].set_title("Processing Time Percentiles")
        axes[0, 2].set_xlabel("Percentile")
        axes[0, 2].set_ylabel("Processing Time (seconds)")
        axes[0, 2].legend()
        
        # Memory usage over time (simulated)
        axes[1, 0].plot(range(len(model_names)), memory_usage, marker='o')
        axes[1, 0].set_title("Memory Usage by Model")
        axes[1, 0].set_xlabel("Model Index")
        axes[1, 0].set_ylabel("Memory Usage (MB)")
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        
        # Performance efficiency (throughput per MB)
        efficiency = [t / m for t, m in zip(throughputs, memory_usage)]
        axes[1, 1].bar(model_names, efficiency)
        axes[1, 1].set_title("Performance Efficiency (Throughput/MB)")
        axes[1, 1].set_ylabel("Efficiency Score")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Hardware utilization
        if self.hardware_info.get("cuda_available"):
            gpu_memory = [self.results[model].get("average_gpu_memory_usage", 0) / (1024 * 1024) for model in model_names]
            axes[1, 2].bar(model_names, gpu_memory)
            axes[1, 2].set_title("GPU Memory Usage")
            axes[1, 2].set_ylabel("GPU Memory (MB)")
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            axes[1, 2].text(0.5, 0.5, "GPU not available", ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title("GPU Memory Usage")
        
        plt.tight_layout()
        plt.savefig(plots_dir / "performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_all_models(self):
        """Evaluate all configured model variants."""
        logger.info("Starting latency evaluation for all models")
        
        # Load test data
        self._load_test_data()
        
        # Evaluate each model variant
        for model_config in self.config["model_variants"]:
            model_path = model_config["path"]
            model_name = model_config["name"]
            model_type = model_config.get("type", "pytorch")
            
            try:
                if model_type == "pytorch":
                    results = self._evaluate_pytorch_model(model_path, model_name)
                elif model_type == "gguf":
                    results = self._evaluate_gguf_model(model_path, model_name)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue
                
                if results:
                    self.results[model_name] = results
                    
                    # Save individual results
                    results_file = Path(self.config["output_dir"]) / f"latency_results_{model_name}.json"
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                continue
        
        # Generate performance report
        performance_report = self._generate_performance_report()
        
        # Save performance report
        report_file = Path(self.config["output_dir"]) / "performance_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(performance_report, f, ensure_ascii=False, indent=2)
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Print summary
        self._print_summary()
        
        logger.info("Latency evaluation completed")
    
    def _print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("SO8T LATENCY EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nHardware Information:")
        print(f"  CPU Count: {self.hardware_info['cpu_count']}")
        print(f"  Memory Total: {self.hardware_info['memory_total'] / (1024**3):.2f} GB")
        print(f"  CUDA Available: {self.hardware_info['cuda_available']}")
        if self.hardware_info['cuda_available']:
            print(f"  GPU Count: {self.hardware_info['gpu_count']}")
            print(f"  GPU Names: {', '.join(self.hardware_info['gpu_names'])}")
        
        print(f"\nModel Performance:")
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Average Processing Time: {results['average_processing_time']:.4f}s")
            print(f"  Median Processing Time: {results['median_processing_time']:.4f}s")
            print(f"  P95 Processing Time: {results['p95_processing_time']:.4f}s")
            print(f"  Throughput: {results['throughput']:.4f} req/s")
            print(f"  Average Memory Usage: {results['average_memory_usage'] / (1024**2):.2f} MB")
            if results.get('average_gpu_memory_usage', 0) > 0:
                print(f"  Average GPU Memory Usage: {results['average_gpu_memory_usage'] / (1024**2):.2f} MB")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate SO8T Model Latency")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--llama_cpp_path", type=str, default="llama-cli", help="Path to llama.cpp executable")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line arguments
    config.update({
        "test_data_path": args.test_data,
        "output_dir": args.output_dir,
        "base_model_name": args.base_model,
        "llama_cpp_path": args.llama_cpp_path,
        "max_length": args.max_length
    })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = LatencyEvaluator(config)
    
    # Run evaluation
    evaluator.evaluate_all_models()


# Example usage
if __name__ == "__main__":
    # Example configuration
    example_config = {
        "test_data_path": "data/test_latency_data.jsonl",
        "output_dir": "latency_results",
        "base_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "llama_cpp_path": "llama-cli",
        "max_length": 512,
        "model_variants": [
            {
                "name": "FP16_LoRA",
                "path": "checkpoints/so8t_qwen2.5-7b_sft_fp16",
                "type": "pytorch"
            },
            {
                "name": "Q4_K_M_GGUF",
                "path": "dist/so8t_qwen2.5-7b-safeagent-q4_k_m.gguf",
                "type": "gguf"
            },
            {
                "name": "Q4_K_S_GGUF",
                "path": "dist/so8t_qwen2.5-7b-safeagent-q4_k_s.gguf",
                "type": "gguf"
            }
        ]
    }
    
    # Create evaluator
    evaluator = LatencyEvaluator(example_config)
    
    # Run evaluation
    evaluator.evaluate_all_models()
    
    # Run main function if called directly
    main()
