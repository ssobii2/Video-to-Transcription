import os
import torch
import psutil
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Environment enum removed - using hardware-based detection only

class ModelSize(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v2"
    LARGE_V3 = "large-v3"

@dataclass
class HardwareInfo:
    """Hardware detection and capability assessment"""
    has_gpu: bool
    gpu_memory_gb: float
    cpu_cores: int
    ram_gb: float
    gpu_name: str = ""
    
    @classmethod
    def detect(cls) -> 'HardwareInfo':
        """Detect current hardware capabilities"""
        has_gpu = torch.cuda.is_available()
        gpu_memory_gb = 0.0
        gpu_name = ""
        
        if has_gpu:
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                has_gpu = False
        
        cpu_cores = psutil.cpu_count(logical=False) or 1
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        return cls(
            has_gpu=has_gpu,
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpu_name=gpu_name
        )

@dataclass
class ModelConfig:
    """Configuration for Whisper model selection"""
    model_size: ModelSize
    device: str
    compute_type: str
    num_workers: int
    batch_size: int
    
    def __post_init__(self):
        # Validate configuration
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            self.compute_type = "int8"

class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Get the absolute path to the project root (where this config.py file is located)
        self.project_root = Path(__file__).parent.absolute()
        
        # Detect hardware and configure models based on hardware only
        self.hardware = HardwareInfo.detect()
        self.model_config = self._get_optimal_model_config()
        
        # Paths - use absolute paths based on project root
        self.input_folder = str(self.project_root / 'input')
        self.output_folder = str(self.project_root / 'output')
        self.transcription_folder = str(self.project_root / 'transcription')
        self.ai_responses_folder = str(self.project_root / 'ai_responses')
        
        # File settings
        self.allowed_extensions = {
            'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', '3gp',
            'mp3', 'wav', 'ogg', 'm4a', 'aac'
        }
        
        # Processing settings
        self.max_file_size_mb = int(os.getenv('MAX_FILE_SIZE_MB', '5000'))  # Increased to 1GB default
        self.chunk_duration = int(os.getenv('CHUNK_DURATION', '30'))
        
        # OpenAI settings
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        
        # Server settings
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', '8000'))
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Ensure directories exist
        self._create_directories()
    
    def _get_downloaded_models(self) -> List[ModelSize]:
        """Check which models are actually downloaded and cached"""
        downloaded_models = []
        
        # Check HuggingFace cache directories
        cache_dirs = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
            Path.home() / ".cache" / "whisper"
        ]
        
        model_mapping = {
            ModelSize.TINY: "tiny",
            ModelSize.BASE: "base", 
            ModelSize.SMALL: "small",
            ModelSize.MEDIUM: "medium",
            ModelSize.LARGE: "large-v2",
            ModelSize.LARGE_V3: "large-v3"
        }
        
        for model_size, model_name in model_mapping.items():
            for cache_dir in cache_dirs:
                if cache_dir and cache_dir.exists():
                    # Look for model directories that match the pattern more specifically
                    # Use more specific patterns to avoid false positives
                    pattern_matches = []
                    
                    # Check for Systran faster-whisper models (most common)
                    pattern_matches.extend(list(cache_dir.glob(f"models--Systran--faster-whisper-{model_name}")))
                    pattern_matches.extend(list(cache_dir.glob(f"models--Systran--faster-whisper-{model_name}.*")))
                    
                    # Check for OpenAI models
                    pattern_matches.extend(list(cache_dir.glob(f"models--openai--whisper-{model_name}")))
                    pattern_matches.extend(list(cache_dir.glob(f"models--openai--whisper-{model_name}.*")))
                    
                    # Only check broader patterns if no specific matches found
                    if not pattern_matches:
                        # Be more careful with broader patterns - ensure model_name is at word boundary
                        for item in cache_dir.iterdir():
                            item_name = item.name.lower()
                            if item.is_dir() and "whisper" in item_name:
                                # Check if the model name appears as a complete word/token
                                import re
                                pattern = rf"\b{re.escape(model_name)}\b"
                                if re.search(pattern, item_name):
                                    pattern_matches.append(item)
                    
                    if pattern_matches:
                        downloaded_models.append(model_size)
                        break
        
        return downloaded_models

    def _get_optimal_model_config(self) -> ModelConfig:
        """Select optimal model configuration based on hardware and downloaded models"""
        
        # Check which models are downloaded
        downloaded_models = self._get_downloaded_models()
        
        # Use GPU config if GPU available, otherwise CPU config
        if self.hardware.has_gpu:
            return self._get_gpu_config(downloaded_models)
        else:
            return self._get_cpu_config(downloaded_models)
    
    def _get_gpu_config(self, downloaded_models: List[ModelSize] = None) -> ModelConfig:
        """GPU-optimized configuration - prioritize accuracy"""
        if downloaded_models is None:
            downloaded_models = []
            
        # Always prioritize accuracy over speed for GPU - Large-v3 is always first choice
        preferred_order = [ModelSize.LARGE_V3, ModelSize.LARGE, ModelSize.MEDIUM, ModelSize.SMALL, ModelSize.BASE, ModelSize.TINY]
        
        # Define config based on GPU memory for performance optimization
        if self.hardware.gpu_memory_gb >= 10:
            # High-end GPU
            config_template = {
                "device": "cuda",
                "compute_type": "float16",
                "num_workers": 2,
                "batch_size": 8
            }
        elif self.hardware.gpu_memory_gb >= 8:
            # Good GPU
            config_template = {
                "device": "cuda",
                "compute_type": "float16",
                "num_workers": 2,
                "batch_size": 6
            }
        elif self.hardware.gpu_memory_gb >= 6:
            # Mid-range GPU
            config_template = {
                "device": "cuda",
                "compute_type": "float16", 
                "num_workers": 1,
                "batch_size": 4
            }
        elif self.hardware.gpu_memory_gb >= 4:
            # Lower-end GPU
            config_template = {
                "device": "cuda",
                "compute_type": "float16",
                "num_workers": 1,
                "batch_size": 2
            }
        else:
            # Very low GPU memory
            config_template = {
                "device": "cuda",
                "compute_type": "float16",
                "num_workers": 1,
                "batch_size": 1
            }
        
        # Find the best downloaded model that matches our preference
        for preferred_model in preferred_order:
            if preferred_model in downloaded_models:
                return ModelConfig(
                    model_size=preferred_model,
                    **config_template
                )
        
        # If no preferred models are downloaded, fall back to the first preference
        # (it will be downloaded automatically when needed)
        return ModelConfig(
            model_size=preferred_order[0],
            **config_template
        )
    
    def _get_cpu_config(self, downloaded_models: List[ModelSize] = None) -> ModelConfig:
        """CPU-optimized configuration - prioritize accuracy based on hardware capability"""
        if downloaded_models is None:
            downloaded_models = []
            
        # Define preferred order based on hardware capability - always prioritize highest accuracy available
        if self.hardware.ram_gb >= 15 and self.hardware.cpu_cores >= 8:
            # High-end CPU: can handle large models, prioritize Large-v3
            preferred_order = [ModelSize.LARGE_V3, ModelSize.LARGE, ModelSize.MEDIUM, ModelSize.SMALL, ModelSize.BASE, ModelSize.TINY]
            config_template = {
                "device": "cpu",
                "compute_type": "int8",
                "num_workers": min(self.hardware.cpu_cores, 6),
                "batch_size": 1
            }
        elif self.hardware.ram_gb >= 7 and self.hardware.cpu_cores >= 4:
            # Mid-range CPU: prioritize Medium as highest practical accuracy
            preferred_order = [ModelSize.MEDIUM, ModelSize.SMALL, ModelSize.BASE, ModelSize.TINY]
            config_template = {
                "device": "cpu",
                "compute_type": "int8",
                "num_workers": min(self.hardware.cpu_cores, 4),
                "batch_size": 1
            }
        else:
            # Limited CPU: prioritize Small as highest practical accuracy
            preferred_order = [ModelSize.SMALL, ModelSize.BASE, ModelSize.TINY, ModelSize.MEDIUM]
            config_template = {
                "device": "cpu",
                "compute_type": "int8", 
                "num_workers": min(self.hardware.cpu_cores, 2),
                "batch_size": 1
            }
        
        # Find the best downloaded model that matches our preference
        for preferred_model in preferred_order:
            if preferred_model in downloaded_models:
                return ModelConfig(
                    model_size=preferred_model,
                    **config_template
                )
        
        # If no preferred models are downloaded, fall back to the first preference
        return ModelConfig(
            model_size=preferred_order[0],
            **config_template
        )
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.input_folder,
            self.output_folder,
            self.transcription_folder,
            self.ai_responses_folder
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the selected model configuration"""
        downloaded_models = self._get_downloaded_models()
        downloaded_model_names = [model.value for model in downloaded_models]
        
        return {
            "model_size": self.model_config.model_size.value,
            "device": self.model_config.device,
            "compute_type": self.model_config.compute_type,
            "downloaded_models": downloaded_model_names,
            "hardware": {
                "has_gpu": self.hardware.has_gpu,
                "gpu_memory_gb": self.hardware.gpu_memory_gb,
                "gpu_name": self.hardware.gpu_name,
                "cpu_cores": self.hardware.cpu_cores,
                "ram_gb": self.hardware.ram_gb,
            }
        }
    
    def get_downloaded_models_info(self) -> List[str]:
        """Get list of downloaded model names"""
        downloaded_models = self._get_downloaded_models()
        return [model.value for model in downloaded_models] 