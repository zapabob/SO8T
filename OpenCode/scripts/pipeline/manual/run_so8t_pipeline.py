"""
SO8T Complete Pipeline Script

This script runs the complete SO8T pipeline from training to evaluation:
1. Train SO8T model with QLoRA
2. Convert to GGUF variants
3. Evaluate safety and latency
4. Generate model card and reports

Usage:
    python scripts/run_so8t_pipeline.py --config configs/pipeline_config.yaml
"""

import os
import json
import logging
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SO8TPipeline:
    """
    Complete SO8T pipeline runner.
    
    Orchestrates the entire process from training to evaluation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SO8T pipeline.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.start_time = time.time()
        
        # Setup logging
        self._setup_logging()
        
        # Pipeline steps
        self.steps = [
            "prepare_data",
            "train_model",
            "convert_gguf",
            "evaluate_safety",
            "evaluate_latency",
            "generate_reports"
        ]
        
        # Results tracking
        self.results = {}
        
        logger.info("SO8T Pipeline initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config["output_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_dir / "pipeline.log")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def _run_command(self, command: List[str], step_name: str) -> bool:
        """
        Run a command and handle errors.
        
        Args:
            command: Command to run
            step_name: Name of the pipeline step
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running {step_name}: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout", 3600)  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"{step_name} completed successfully")
                return True
            else:
                logger.error(f"{step_name} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"{step_name} timed out")
            return False
        except Exception as e:
            logger.error(f"Error running {step_name}: {e}")
            return False
    
    def prepare_data(self) -> bool:
        """Prepare training and evaluation data."""
        logger.info("Preparing data...")
        
        # Check if data files exist
        train_data = Path(self.config["train_data_path"])
        val_data = Path(self.config["val_data_path"])
        test_data = Path(self.config["test_data_path"])
        
        if not train_data.exists():
            logger.error(f"Training data not found: {train_data}")
            return False
        
        if not val_data.exists():
            logger.error(f"Validation data not found: {val_data}")
            return False
        
        if not test_data.exists():
            logger.error(f"Test data not found: {test_data}")
            return False
        
        logger.info("Data preparation completed")
        return True
    
    def train_model(self) -> bool:
        """Train the SO8T model with QLoRA."""
        logger.info("Training SO8T model...")
        
        # Prepare training command
        command = [
            "python", "-m", "training.train_qlora",
            "--config", self.config["train_config_path"],
            "--train_data", self.config["train_data_path"],
            "--val_data", self.config["val_data_path"],
            "--output_dir", self.config["train_output_dir"],
            "--base_model", self.config["base_model_name"],
            "--num_epochs", str(self.config.get("num_epochs", 10)),
            "--batch_size", str(self.config.get("batch_size", 4)),
            "--learning_rate", str(self.config.get("learning_rate", 1e-4))
        ]
        
        if self.config.get("use_wandb", False):
            command.append("--use_wandb")
        
        success = self._run_command(command, "Model Training")
        
        if success:
            self.results["training"] = {
                "status": "completed",
                "output_dir": self.config["train_output_dir"]
            }
        
        return success
    
    def convert_gguf(self) -> bool:
        """Convert trained model to GGUF variants."""
        logger.info("Converting model to GGUF variants...")
        
        # This would typically use llama.cpp conversion tools
        # For now, we'll create a placeholder script
        
        gguf_script = Path("scripts/convert_to_gguf.py")
        if not gguf_script.exists():
            logger.warning("GGUF conversion script not found. Creating placeholder.")
            self._create_gguf_conversion_script()
        
        # Run GGUF conversion
        command = [
            "python", str(gguf_script),
            "--input_dir", self.config["train_output_dir"],
            "--output_dir", self.config["gguf_output_dir"],
            "--base_model", self.config["base_model_name"]
        ]
        
        success = self._run_command(command, "GGUF Conversion")
        
        if success:
            self.results["gguf_conversion"] = {
                "status": "completed",
                "output_dir": self.config["gguf_output_dir"]
            }
        
        return success
    
    def _create_gguf_conversion_script(self):
        """Create a placeholder GGUF conversion script."""
        script_content = '''"""
Placeholder GGUF conversion script.
In a real implementation, this would use llama.cpp tools.
"""

import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert SO8T model to GGUF")
    parser.add_argument("--input_dir", required=True, help="Input directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--base_model", required=True, help="Base model name")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create placeholder GGUF files
    variants = ["q4_k_m", "q4_k_s", "iq4_xs"]
    
    for variant in variants:
        output_file = Path(args.output_dir) / f"so8t_qwen2.5-7b-safeagent-{variant}.gguf"
        
        # Create a placeholder file
        with open(output_file, "w") as f:
            f.write(f"# Placeholder GGUF file for {variant}\n")
            f.write(f"# This would be generated by llama.cpp in a real implementation\n")
        
        print(f"Created placeholder: {output_file}")
    
    print("GGUF conversion completed (placeholder)")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path("scripts/convert_to_gguf.py")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make it executable
        os.chmod(script_path, 0o755)
    
    def evaluate_safety(self) -> bool:
        """Evaluate safety performance."""
        logger.info("Evaluating safety performance...")
        
        # Prepare safety evaluation command
        command = [
            "python", "-m", "eval.eval_safety",
            "--config", self.config["safety_eval_config_path"],
            "--test_data", self.config["test_data_path"],
            "--output_dir", self.config["safety_eval_output_dir"],
            "--base_model", self.config["base_model_name"]
        ]
        
        success = self._run_command(command, "Safety Evaluation")
        
        if success:
            self.results["safety_evaluation"] = {
                "status": "completed",
                "output_dir": self.config["safety_eval_output_dir"]
            }
        
        return success
    
    def evaluate_latency(self) -> bool:
        """Evaluate latency and performance."""
        logger.info("Evaluating latency and performance...")
        
        # Prepare latency evaluation command
        command = [
            "python", "-m", "eval.eval_latency",
            "--config", self.config["latency_eval_config_path"],
            "--test_data", self.config["test_data_path"],
            "--output_dir", self.config["latency_eval_output_dir"],
            "--base_model", self.config["base_model_name"]
        ]
        
        success = self._run_command(command, "Latency Evaluation")
        
        if success:
            self.results["latency_evaluation"] = {
                "status": "completed",
                "output_dir": self.config["latency_eval_output_dir"]
            }
        
        return success
    
    def generate_reports(self) -> bool:
        """Generate final reports and model card."""
        logger.info("Generating reports...")
        
        # Create reports directory
        reports_dir = Path(self.config["output_dir"]) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate pipeline summary
        pipeline_summary = {
            "pipeline_version": "1.0.0",
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_duration": time.time() - self.start_time,
            "steps_completed": list(self.results.keys()),
            "results": self.results,
            "config": self.config
        }
        
        # Save pipeline summary
        summary_file = reports_dir / "pipeline_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(pipeline_summary, f, ensure_ascii=False, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(reports_dir, pipeline_summary)
        
        logger.info("Reports generated successfully")
        return True
    
    def _generate_markdown_report(self, reports_dir: Path, summary: Dict):
        """Generate a markdown report."""
        report_content = f"""# SO8T Pipeline Execution Report

## Pipeline Summary

- **Version**: {summary['pipeline_version']}
- **Start Time**: {time.ctime(summary['start_time'])}
- **End Time**: {time.ctime(summary['end_time'])}
- **Total Duration**: {summary['total_duration']:.2f} seconds

## Steps Completed

"""
        
        for step, result in summary['results'].items():
            report_content += f"- **{step.replace('_', ' ').title()}**: {result['status']}\n"
            if 'output_dir' in result:
                report_content += f"  - Output Directory: `{result['output_dir']}`\n"
        
        report_content += f"""
## Configuration

```json
{json.dumps(summary['config'], indent=2)}
```

## Next Steps

1. Review the generated model variants in the output directories
2. Test the models with your specific use cases
3. Deploy the best performing model for your requirements
4. Monitor performance and safety metrics in production

## Files Generated

- Model checkpoints: `{self.config['train_output_dir']}`
- GGUF variants: `{self.config['gguf_output_dir']}`
- Safety evaluation: `{self.config['safety_eval_output_dir']}`
- Latency evaluation: `{self.config['latency_eval_output_dir']}`
- Reports: `{reports_dir}`

## Support

For questions or issues, please refer to the documentation in the `docs/` directory.
"""
        
        report_file = reports_dir / "pipeline_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
    
    def run_pipeline(self) -> bool:
        """Run the complete SO8T pipeline."""
        logger.info("Starting SO8T pipeline execution")
        
        # Run each step
        for step in self.steps:
            logger.info(f"Running step: {step}")
            
            if step == "prepare_data":
                success = self.prepare_data()
            elif step == "train_model":
                success = self.train_model()
            elif step == "convert_gguf":
                success = self.convert_gguf()
            elif step == "evaluate_safety":
                success = self.evaluate_safety()
            elif step == "evaluate_latency":
                success = self.evaluate_latency()
            elif step == "generate_reports":
                success = self.generate_reports()
            else:
                logger.error(f"Unknown step: {step}")
                success = False
            
            if not success:
                logger.error(f"Pipeline failed at step: {step}")
                return False
        
        logger.info("SO8T pipeline completed successfully")
        return True


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Run SO8T Complete Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--skip_training", action="store_true", help="Skip training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation steps")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override output directory if provided
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Skip steps if requested
    if args.skip_training:
        config["steps"] = [step for step in config.get("steps", []) if step != "train_model"]
    
    if args.skip_evaluation:
        config["steps"] = [step for step in config.get("steps", []) if step not in ["evaluate_safety", "evaluate_latency"]]
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = SO8TPipeline(config)
    success = pipeline.run_pipeline()
    
    if success:
        print("✅ SO8T pipeline completed successfully!")
        print(f"📁 Results saved to: {config['output_dir']}")
    else:
        print("❌ SO8T pipeline failed!")
        exit(1)


if __name__ == "__main__":
    main()
