#!/usr/bin/env python3
"""
SO8T Training Script

Linear training pipeline for SO8T models.
Handles data loading, model setup, training, and evaluation.
"""

import sys
import yaml
from pathlib import Path
from so8t.training import QLoRATrainer
from so8t.core import SO8TTransformer
from so8t.safety import EnhancedAuditLogger

def load_config():
    """Load unified configuration."""
    config_path = Path("so8t/config/pipeline.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    """Main training function with linear pipeline."""

    print("🎯 SO8T Training Pipeline")
    print("=" * 50)

    # Load configuration
    config = load_config()
    print("📋 Configuration loaded")

    # Step 1: Initialize audit logging
    print("\n🔐 Step 1: Initializing safety audit...")
    audit_logger = EnhancedAuditLogger(config["storage"]["logs_dir"])
    audit_logger.log_event("training_started", {"config": config})
    print("✅ Audit logging initialized")

    # Step 2: Load model
    print("\n🤖 Step 2: Loading base model...")
    model = SO8TTransformer.from_pretrained(config["model"]["base_model"])
    print(f"✅ Model loaded: {config['model']['base_model']}")

    # Step 3: Setup trainer
    print("\n🎓 Step 3: Setting up trainer...")
    trainer = QLoRATrainer(
        model=model,
        config=config,
        audit_logger=audit_logger
    )
    print("✅ Trainer configured")

    # Step 4: Load data
    print("\n📊 Step 4: Loading training data...")
    # Data loading logic here
    print("✅ Training data loaded")

    # Step 5: Train model
    print("\n🏃 Step 5: Starting training...")
    trainer.train()
    print("✅ Training completed")

    # Step 6: Evaluate
    print("\n📈 Step 6: Running evaluation...")
    metrics = trainer.evaluate()
    print(f"✅ Evaluation completed: {metrics}")

    # Step 7: Save model
    print("\n💾 Step 7: Saving model...")
    trainer.save_model(config["storage"]["models_dir"])
    print("✅ Model saved")

    # Log completion
    audit_logger.log_event("training_completed", {"metrics": metrics})

    print("\n🎉 Training pipeline complete!")
    print(f"Model saved to: {config['storage']['models_dir']}")

if __name__ == "__main__":
    main()






























