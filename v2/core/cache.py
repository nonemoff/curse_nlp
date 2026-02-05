"""Cache manager for storing intermediate results."""
import pickle
from pathlib import Path

import config


class CacheManager:
    """Manages caching of analysis results."""
    
    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
    
    def save(self, name: str, data: dict):
        """Save data to cache."""
        filepath = self.cache_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, name: str):
        """Load data from cache."""
        filepath = self.cache_dir / f"{name}.pkl"
        if not filepath.exists():
            return None
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def exists(self, name: str) -> bool:
        """Check if cache exists."""
        filepath = self.cache_dir / f"{name}.pkl"
        return filepath.exists()
    
    def clear_all(self):
        """Clear all cache files."""
        for filepath in self.cache_dir.glob("*.pkl"):
            filepath.unlink()
    
    def get_status(self) -> dict:
        """Get cache status for all modules."""
        modules = ['extracted', 'frequency', 'terms', 'names']
        status = {}
        
        for module in modules:
            filepath = self.cache_dir / f"{module}.pkl"
            status[module] = {
                'exists': filepath.exists(),
                'size': f"{filepath.stat().st_size / 1024:.1f} KB" if filepath.exists() else None
            }
        
        return status
