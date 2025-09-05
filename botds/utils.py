"""Utility functions for hashing, time, and I/O operations."""

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd


def generate_job_id() -> str:
    """Generate a unique 8-character job ID."""
    return str(uuid.uuid4())[:8]


def get_timestamp() -> str:
    """Get current timestamp in ISO8601 format."""
    return datetime.utcnow().isoformat() + "Z"


def hash_object(obj: Any) -> str:
    """Generate SHA256 hash of an object."""
    if isinstance(obj, pd.DataFrame):
        # Hash DataFrame content
        content = obj.to_csv(index=False).encode()
    elif isinstance(obj, np.ndarray):
        content = obj.tobytes()
    elif isinstance(obj, (dict, list)):
        content = json.dumps(obj, sort_keys=True).encode()
    elif isinstance(obj, str):
        content = obj.encode()
    elif isinstance(obj, Path):
        # Hash file content if it exists
        if obj.exists():
            content = obj.read_bytes()
        else:
            content = str(obj).encode()
    else:
        content = str(obj).encode()
    
    return "sha256:" + hashlib.sha256(content).hexdigest()


def hash_dataset(df: pd.DataFrame, name: str = "") -> str:
    """Generate a comprehensive hash for a dataset."""
    components = [
        name,
        str(df.shape),
        str(df.dtypes.to_dict()),
        str(df.columns.tolist()),
        hash_object(df.head(100)),  # Sample for large datasets
        hash_object(df.tail(100)),
    ]
    combined = "|".join(components)
    return hash_object(combined)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Union[str, Path]) -> str:
    """Save object as JSON and return hash."""
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    
    return hash_object(obj)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON from file."""
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> str:
    """Save object as pickle and return hash."""
    path = Path(path)
    ensure_dir(path.parent)
    
    joblib.dump(obj, path)
    return hash_object(obj)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickle from file."""
    return joblib.load(path)


def get_memory_usage_gb() -> float:
    """Get current memory usage in GB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return f"{self.name}: {format_duration(self.elapsed)}"
