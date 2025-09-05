"""File cache system with intelligent invalidation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .utils import ensure_dir, hash_object, load_json, load_pickle, save_json, save_pickle


class CacheIndex:
    """Index for tracking cache entries and dependencies."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = ensure_dir(cache_dir)
        self.index_path = self.cache_dir / "index.json"
        self.index: Dict[str, Dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_path.exists():
            return load_json(self.index_path)
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        save_json(self.index, self.index_path)
    
    def get_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cache entry by key."""
        return self.index.get(key)
    
    def put_entry(
        self,
        key: str,
        file_path: str,
        dependencies: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add cache entry."""
        self.index[key] = {
            "file_path": file_path,
            "dependencies": dependencies,
            "metadata": metadata or {},
            "created_at": hash_object("timestamp")  # Simple timestamp
        }
        self._save_index()
    
    def invalidate_key(self, key: str) -> None:
        """Remove cache entry."""
        if key in self.index:
            # Remove file if it exists
            entry = self.index[key]
            file_path = Path(entry["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            del self.index[key]
            self._save_index()
    
    def invalidate_dependents(self, changed_key: str) -> Set[str]:
        """Invalidate all entries that depend on changed_key."""
        invalidated = set()
        
        for key, entry in list(self.index.items()):
            if changed_key in entry.get("dependencies", []):
                self.invalidate_key(key)
                invalidated.add(key)
        
        return invalidated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.index),
            "cache_dir": str(self.cache_dir),
            "index_path": str(self.index_path)
        }


class Cache:
    """Main cache interface."""
    
    def __init__(self, cache_dir: str, mode: str = "warm"):
        self.cache_dir = Path(cache_dir)
        self.mode = mode  # warm, cold, paranoid
        self.index = CacheIndex(self.cache_dir)
        self.hits: Dict[str, bool] = {}
    
    def _get_cache_path(self, stage: str, key: str) -> Path:
        """Get cache file path for stage and key."""
        return self.cache_dir / stage / f"{key}.pkl"
    
    def get(self, stage: str, key: str) -> Optional[Any]:
        """Get cached object."""
        if self.mode == "cold":
            self.hits[f"{stage}:{key}"] = False
            return None
        
        cache_key = f"{stage}:{key}"
        entry = self.index.get_entry(cache_key)
        
        if entry is None:
            self.hits[cache_key] = False
            return None
        
        cache_path = Path(entry["file_path"])
        if not cache_path.exists():
            self.index.invalidate_key(cache_key)
            self.hits[cache_key] = False
            return None
        
        try:
            obj = load_pickle(cache_path)
            self.hits[cache_key] = True
            return obj
        except Exception:
            self.index.invalidate_key(cache_key)
            self.hits[cache_key] = False
            return None
    
    def put(
        self,
        stage: str,
        key: str,
        obj: Any,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Cache an object."""
        if self.mode == "cold":
            return
        
        cache_path = self._get_cache_path(stage, key)
        ensure_dir(cache_path.parent)
        
        # Save object
        save_pickle(obj, cache_path)
        
        # Update index
        cache_key = f"{stage}:{key}"
        self.index.put_entry(
            cache_key,
            str(cache_path),
            dependencies or [],
            {"stage": stage, "key": key}
        )
    
    def invalidate_stage(self, stage: str) -> None:
        """Invalidate all cache entries for a stage."""
        stage_keys = [k for k in self.index.index.keys() if k.startswith(f"{stage}:")]
        for key in stage_keys:
            self.index.invalidate_key(key)
    
    def invalidate_downstream(self, changed_stage: str) -> Set[str]:
        """Invalidate stages downstream of changed stage."""
        # Define stage dependencies
        stage_deps = {
            "profile": [],
            "eda": ["profile"],
            "feature_plan": ["profile", "eda"],
            "split_indices": ["profile"],
            "ladder": ["feature_plan", "split_indices"],
            "evaluation": ["ladder"],
            "reports": ["evaluation"]
        }
        
        invalidated = set()
        
        # Find all stages that depend on changed_stage
        for stage, deps in stage_deps.items():
            if changed_stage in deps:
                self.invalidate_stage(stage)
                invalidated.add(stage)
                # Recursively invalidate downstream
                invalidated.update(self.invalidate_downstream(stage))
        
        return invalidated
    
    def get_hit_stats(self) -> Dict[str, bool]:
        """Get cache hit/miss statistics."""
        return self.hits.copy()
    
    def clear_all(self) -> None:
        """Clear entire cache."""
        for key in list(self.index.index.keys()):
            self.index.invalidate_key(key)
