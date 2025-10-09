# Configuration management

from pathlib import Path
from typing import Dict
import yaml

class Config:
    """Global configuration"""
    
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
    
    @property
    def gamma_dir(self) -> Path:
        return Path(self._config['paths']['gamma_dir'])
    
    @property
    def plot_defaults(self) -> Dict:
        return self._config['plotting']
    
    @property
    def kernel_size(self) -> int:
        return self._config['analysis']['kernel_size']

# Global config instance
CONFIG = Config('config.yaml')