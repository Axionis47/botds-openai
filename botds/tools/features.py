"""Feature engineering and data splitting tools."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..utils import load_pickle, save_json, save_pickle


class Splitter:
    """Data splitting utilities."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.handoffs_dir = artifacts_dir / "handoffs"
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)
    
    def make_splits(
        self,
        df_ref: str,
        target: str,
        policy: str,
        test_size: float = 0.2,
        val_size: float = 0.2,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Create train/validation/test splits.
        
        Args:
            df_ref: Reference to DataFrame
            target: Target column name
            policy: 'iid' or 'time'
            test_size: Test set proportion
            val_size: Validation set proportion
            seed: Random seed
            
        Returns:
            Split indices saved to split_indices.json
        """
        df = load_pickle(df_ref)
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        
        n_samples = len(df)
        indices = np.arange(n_samples)
        
        if policy == "iid":
            # Random stratified splits
            y = df[target]
            
            # First split: train+val vs test
            train_val_idx, test_idx = train_test_split(
                indices, 
                test_size=test_size,
                stratify=y if y.nunique() <= 20 else None,  # Stratify if classification
                random_state=seed
            )
            
            # Second split: train vs val
            y_train_val = y.iloc[train_val_idx]
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size / (1 - test_size),  # Adjust for remaining data
                stratify=y_train_val if y_train_val.nunique() <= 20 else None,
                random_state=seed
            )
            
        elif policy == "time":
            # Temporal splits (assumes data is sorted by time)
            test_start = int(n_samples * (1 - test_size))
            val_start = int(test_start * (1 - val_size))
            
            train_idx = indices[:val_start]
            val_idx = indices[val_start:test_start]
            test_idx = indices[test_start:]
            
        else:
            raise ValueError(f"Unknown split policy: {policy}")
        
        # Create split info
        split_info = {
            "policy": policy,
            "seed": seed,
            "sizes": {
                "train": len(train_idx),
                "val": len(val_idx), 
                "test": len(test_idx)
            },
            "proportions": {
                "train": len(train_idx) / n_samples,
                "val": len(val_idx) / n_samples,
                "test": len(test_idx) / n_samples
            },
            "indices": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist()
            }
        }
        
        # Add target distribution info
        y = df[target]
        if y.nunique() <= 20:  # Classification
            for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
                y_split = y.iloc[idx]
                split_info[f"{split_name}_target_dist"] = y_split.value_counts().to_dict()
        
        # Save splits
        splits_path = self.handoffs_dir / "split_indices.json"
        hash_value = save_json(split_info, splits_path)
        
        return {
            "splits_ref": str(splits_path),
            "hash": hash_value,
            "summary": split_info["sizes"]
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Splitter_make_splits",
                    "description": "Create train/validation/test splits using IID or temporal policy",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df_ref": {
                                "type": "string",
                                "description": "Reference to pickled DataFrame"
                            },
                            "target": {
                                "type": "string",
                                "description": "Target column name"
                            },
                            "policy": {
                                "type": "string",
                                "enum": ["iid", "time"],
                                "description": "Splitting policy"
                            },
                            "test_size": {
                                "type": "number",
                                "description": "Test set proportion (default: 0.2)"
                            },
                            "val_size": {
                                "type": "number", 
                                "description": "Validation set proportion (default: 0.2)"
                            },
                            "seed": {
                                "type": "integer",
                                "description": "Random seed (default: 42)"
                            }
                        },
                        "required": ["df_ref", "target", "policy"]
                    }
                }
            }
        ]


class Featurizer:
    """Feature engineering and preprocessing."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.handoffs_dir = artifacts_dir / "handoffs"
        self.data_dir = artifacts_dir / "data"
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def plan(
        self,
        df_ref: str,
        target: str,
        deny_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create feature engineering plan.
        
        Args:
            df_ref: Reference to DataFrame
            target: Target column name
            deny_list: Columns to exclude
            
        Returns:
            Feature plan saved to feature_plan.json
        """
        df = load_pickle(df_ref)
        deny_list = deny_list or []
        
        # Exclude target and denied columns
        feature_cols = [col for col in df.columns if col != target and col not in deny_list]
        
        plan = {
            "target": target,
            "feature_columns": feature_cols,
            "denied_columns": deny_list,
            "transforms": []
        }
        
        # Analyze each feature column
        for col in feature_cols:
            col_info = {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique())
            }
            
            # Determine transforms needed
            transforms = []
            
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    transforms.append("impute_median")
                else:
                    transforms.append("impute_mode")
            
            if df[col].dtype in ['object', 'category']:
                if df[col].nunique() > 50:
                    transforms.append("high_cardinality_encode")
                else:
                    transforms.append("label_encode")
            elif df[col].dtype in ['int64', 'float64']:
                transforms.append("standard_scale")
            
            col_info["transforms"] = transforms
            plan["transforms"].append(col_info)
        
        # Add rationale
        plan["rationale"] = {
            "total_features": len(feature_cols),
            "excluded_features": len(deny_list),
            "numeric_features": len([c for c in feature_cols if df[c].dtype in ['int64', 'float64']]),
            "categorical_features": len([c for c in feature_cols if df[c].dtype in ['object', 'category']]),
            "missing_data_columns": len([c for c in feature_cols if df[c].isnull().sum() > 0])
        }
        
        # Save plan
        plan_path = self.handoffs_dir / "feature_plan.json"
        hash_value = save_json(plan, plan_path)
        
        return {
            "plan_ref": str(plan_path),
            "hash": hash_value,
            "summary": plan["rationale"]
        }
    
    def apply(
        self,
        df_ref: str,
        plan_ref: str,
        splits_ref: str
    ) -> Dict[str, Any]:
        """
        Apply feature engineering plan to create train/val/test matrices.
        
        Args:
            df_ref: Reference to DataFrame
            plan_ref: Reference to feature plan
            splits_ref: Reference to split indices
            
        Returns:
            References to processed X,y matrices
        """
        from ..utils import load_json
        
        df = load_pickle(df_ref)
        plan = load_json(plan_ref)
        splits = load_json(splits_ref)
        
        target = plan["target"]
        feature_cols = plan["feature_columns"]
        
        # Get splits
        train_idx = splits["indices"]["train"]
        val_idx = splits["indices"]["val"]
        test_idx = splits["indices"]["test"]
        
        # Prepare features and target
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Apply transforms (fit on train, transform all)
        X_processed = X.copy()
        
        # Simple preprocessing for MVP
        for col in feature_cols:
            if X[col].dtype in ['object', 'category']:
                # Label encoding
                le = LabelEncoder()
                X_train_col = X.iloc[train_idx][col].fillna('missing')
                le.fit(X_train_col)
                
                X_processed[col] = X[col].fillna('missing').apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            elif X[col].dtype in ['int64', 'float64']:
                # Fill missing with median and scale
                median_val = X.iloc[train_idx][col].median()
                X_processed[col] = X[col].fillna(median_val)
                
                # Standard scaling
                scaler = StandardScaler()
                X_train_col = X_processed.iloc[train_idx][[col]]
                scaler.fit(X_train_col)
                X_processed[col] = scaler.transform(X_processed[[col]]).flatten()
        
        # Create train/val/test sets
        X_train = X_processed.iloc[train_idx]
        X_val = X_processed.iloc[val_idx]
        X_test = X_processed.iloc[test_idx]
        
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_test = y.iloc[test_idx]
        
        # Save matrices
        matrices = {}
        for name, (X_split, y_split) in [
            ("train", (X_train, y_train)),
            ("val", (X_val, y_val)),
            ("test", (X_test, y_test))
        ]:
            X_path = self.data_dir / f"X_{name}.pkl"
            y_path = self.data_dir / f"y_{name}.pkl"
            
            save_pickle(X_split, X_path)
            save_pickle(y_split, y_path)
            
            matrices[f"X_{name}_ref"] = str(X_path)
            matrices[f"y_{name}_ref"] = str(y_path)
        
        return matrices
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Featurizer_plan",
                    "description": "Create feature engineering plan with transforms and rationale",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df_ref": {
                                "type": "string",
                                "description": "Reference to pickled DataFrame"
                            },
                            "target": {
                                "type": "string",
                                "description": "Target column name"
                            },
                            "deny_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to exclude from features"
                            }
                        },
                        "required": ["df_ref", "target"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Featurizer_apply",
                    "description": "Apply feature engineering plan to create processed train/val/test matrices",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df_ref": {
                                "type": "string",
                                "description": "Reference to pickled DataFrame"
                            },
                            "plan_ref": {
                                "type": "string",
                                "description": "Reference to feature plan JSON"
                            },
                            "splits_ref": {
                                "type": "string",
                                "description": "Reference to split indices JSON"
                            }
                        },
                        "required": ["df_ref", "plan_ref", "splits_ref"]
                    }
                }
            }
        ]
