"""
SO8T Distilled Model Loader

This module provides functionality to load and prepare SO8T distilled models
for GGUF conversion. It handles the PyTorch checkpoint format and validates
SO8T-specific structure.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.so8t_model import SO8TModel, SO8TModelConfig

logger = logging.getLogger(__name__)


class SO8TDistilledModelLoader:
    """
    SO8T Distilled Model Loader
    
    Features:
    - Load PyTorch checkpoints (.pt files)
    - Validate SO8T structure
    - Extract model configuration
    - Prepare for GGUF conversion
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Initialize model loader
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to load model on
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.checkpoint = None
        self.model_config = None
        self.state_dict = None
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load PyTorch checkpoint
        
        Returns:
            Dictionary containing checkpoint data
        """
        try:
            logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
            
            # Load with map_location to avoid GPU requirements
            self.checkpoint = torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False  # Allow loading optimizer state etc.
            )
            
            logger.info("Checkpoint loaded successfully")
            return self.checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def validate_so8t_structure(self) -> bool:
        """
        Validate SO8T model structure in checkpoint
        
        Returns:
            True if valid SO8T structure, False otherwise
        """
        try:
            if self.checkpoint is None:
                self.load_checkpoint()
            
            # Check for required SO8T components
            required_components = [
                'group_structure',  # SO(8) group structure
                'task_head_a',      # Task reasoning head
                'safety_head_b',    # Safety reasoning head
            ]
            
            # Get state dict (might be nested)
            if 'model_state_dict' in self.checkpoint:
                state_dict = self.checkpoint['model_state_dict']
            elif 'state_dict' in self.checkpoint:
                state_dict = self.checkpoint['state_dict']
            else:
                state_dict = self.checkpoint
            
            self.state_dict = state_dict
            
            # Check for SO8T components
            missing_components = []
            for component in required_components:
                found = any(component in key for key in state_dict.keys())
                if not found:
                    missing_components.append(component)
            
            if missing_components:
                logger.warning(f"Missing SO8T components: {missing_components}")
                logger.info("This may be a standard model without SO8T structure")
                return False
            
            logger.info("SO8T structure validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate SO8T structure: {e}")
            return False
    
    def extract_config(self) -> Dict[str, Any]:
        """
        Extract model configuration from checkpoint
        
        Returns:
            Dictionary containing model configuration
        """
        try:
            if self.checkpoint is None:
                self.load_checkpoint()
            
            # Try to find config in checkpoint
            if 'config' in self.checkpoint:
                config = self.checkpoint['config']
            elif 'model_config' in self.checkpoint:
                config = self.checkpoint['model_config']
            elif 'hparams' in self.checkpoint:
                config = self.checkpoint['hparams']
            else:
                # Create default config
                logger.warning("No config found in checkpoint, using defaults")
                config = {
                    'base_model_name': 'Qwen/Qwen2.5-7B-Instruct',
                    'task_head_hidden_size': 4096,
                    'safety_head_hidden_size': 2048,
                    'safety_num_classes': 3,
                    'vocab_size': 151936,
                }
            
            self.model_config = config
            logger.info(f"Extracted config: {json.dumps(config, indent=2)}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to extract config: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary containing model info
        """
        try:
            if self.checkpoint is None:
                self.load_checkpoint()
            
            if self.state_dict is None:
                self.validate_so8t_structure()
            
            # Count parameters
            total_params = sum(
                p.numel() for p in self.state_dict.values() 
                if isinstance(p, torch.Tensor)
            )
            
            # Get tensor info
            tensor_info = {}
            for name, tensor in self.state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_info[name] = {
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'device': str(tensor.device),
                        'requires_grad': tensor.requires_grad,
                    }
            
            model_info = {
                'checkpoint_path': str(self.checkpoint_path),
                'total_parameters': total_params,
                'parameter_count_millions': round(total_params / 1e6, 2),
                'num_tensors': len(tensor_info),
                'config': self.model_config,
                'has_so8t_structure': self.validate_so8t_structure(),
                'tensor_names': list(tensor_info.keys()),
            }
            
            # Check for training info
            if 'epoch' in self.checkpoint:
                model_info['training_epoch'] = self.checkpoint['epoch']
            if 'optimizer_state_dict' in self.checkpoint:
                model_info['has_optimizer_state'] = True
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def prepare_for_conversion(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Prepare model for GGUF conversion
        
        Returns:
            Tuple of (state_dict, config)
        """
        try:
            if self.checkpoint is None:
                self.load_checkpoint()
            
            if self.state_dict is None:
                self.validate_so8t_structure()
            
            if self.model_config is None:
                self.extract_config()
            
            # Convert all tensors to CPU and float32 for compatibility
            prepared_state_dict = {}
            for name, tensor in self.state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    # Move to CPU and convert to float32 if needed
                    prepared_tensor = tensor.cpu()
                    if prepared_tensor.dtype in [torch.float16, torch.bfloat16]:
                        prepared_tensor = prepared_tensor.float()
                    prepared_state_dict[name] = prepared_tensor
            
            logger.info(f"Prepared {len(prepared_state_dict)} tensors for conversion")
            
            return prepared_state_dict, self.model_config
            
        except Exception as e:
            logger.error(f"Failed to prepare for conversion: {e}")
            raise
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save model configuration to JSON file
        
        Args:
            output_path: Path to save config (default: same dir as checkpoint)
        """
        try:
            if self.model_config is None:
                self.extract_config()
            
            if output_path is None:
                output_path = self.checkpoint_path.parent / f"{self.checkpoint_path.stem}_config.json"
            else:
                output_path = Path(output_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Config saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise


def load_distilled_model(pt_path: str, device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Convenience function to load distilled model
    
    Args:
        pt_path: Path to .pt checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (state_dict, config)
    """
    loader = SO8TDistilledModelLoader(pt_path, device)
    return loader.prepare_for_conversion()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Load distilled model
    checkpoint_path = "models/so8t_distilled_safety.pt"
    
    try:
        loader = SO8TDistilledModelLoader(checkpoint_path)
        
        # Load and validate
        loader.load_checkpoint()
        is_valid = loader.validate_so8t_structure()
        
        if is_valid:
            logger.info("SO8T structure validated")
        else:
            logger.warning("Not a standard SO8T model")
        
        # Get model info
        model_info = loader.get_model_info()
        print("\nModel Information:")
        print(json.dumps(model_info, indent=2, default=str))
        
        # Prepare for conversion
        state_dict, config = loader.prepare_for_conversion()
        print(f"\nPrepared {len(state_dict)} tensors for GGUF conversion")
        
        # Save config
        loader.save_config()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

