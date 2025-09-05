"""Data I/O tools for loading datasets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn import datasets

from ..utils import ensure_dir, hash_dataset, save_pickle


class DataStore:
    """Data loading and management tools."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.data_dir = ensure_dir(artifacts_dir / "data")
    
    def read_builtin(self, name: str) -> Dict[str, Any]:
        """
        Load a built-in sklearn dataset.
        
        Args:
            name: Dataset name (iris, breast_cancer, diabetes)
            
        Returns:
            Dict with df_ref, target, task_hint
        """
        if name == "iris":
            data = datasets.load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
            task_hint = 'classification'
            
        elif name == "breast_cancer":
            data = datasets.load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
            task_hint = 'classification'
            
        elif name == "diabetes":
            data = datasets.load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            target = 'target'
            task_hint = 'regression'
            
        else:
            raise ValueError(f"Unknown builtin dataset: {name}")
        
        # Save dataset
        dataset_hash = hash_dataset(df, name)
        df_path = self.data_dir / f"{name}_{dataset_hash[:8]}.pkl"
        save_pickle(df, df_path)
        
        return {
            "df_ref": str(df_path),
            "target": target,
            "task_hint": task_hint,
            "shape": df.shape,
            "hash": dataset_hash
        }
    
    def read_csv(self, paths: List[str]) -> Dict[str, Any]:
        """
        Load CSV files and combine if multiple.
        
        Args:
            paths: List of CSV file paths
            
        Returns:
            Dict with df_ref
        """
        if not paths:
            raise ValueError("No CSV paths provided")
        
        dfs = []
        for path in paths:
            csv_path = Path(path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            
            df = pd.read_csv(csv_path)
            dfs.append(df)
        
        # Combine if multiple files
        if len(dfs) == 1:
            combined_df = dfs[0]
        else:
            combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save combined dataset
        dataset_hash = hash_dataset(combined_df, "csv")
        df_path = self.data_dir / f"csv_{dataset_hash[:8]}.pkl"
        save_pickle(combined_df, df_path)
        
        return {
            "df_ref": str(df_path),
            "shape": combined_df.shape,
            "hash": dataset_hash
        }
    
    def load_dataframe(self, df_ref: str) -> pd.DataFrame:
        """Load DataFrame from reference."""
        from ..utils import load_pickle
        return load_pickle(df_ref)
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions for this tool."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "DataStore_read_builtin",
                    "description": "Load a built-in sklearn dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["iris", "breast_cancer", "diabetes"],
                                "description": "Name of the built-in dataset"
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "DataStore_read_csv",
                    "description": "Load CSV files and combine if multiple",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of CSV file paths to load"
                            }
                        },
                        "required": ["paths"]
                    }
                }
            }
        ]
