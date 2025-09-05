"""Data profiling and quality assessment tools."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import load_pickle, save_json


class SchemaProfiler:
    """Dataset schema and basic profiling."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.handoffs_dir = artifacts_dir / "handoffs"
        self.handoffs_dir.mkdir(parents=True, exist_ok=True)
    
    def profile(self, df_ref: str) -> Dict[str, Any]:
        """
        Generate comprehensive dataset profile.
        
        Args:
            df_ref: Reference to pickled DataFrame
            
        Returns:
            Profile dictionary saved to profile.json
        """
        df = load_pickle(df_ref)
        
        # Basic shape and structure
        profile = {
            "shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Missing values analysis
        missing = df.isnull().sum()
        profile["missing_values"] = {
            "total_missing": int(missing.sum()),
            "missing_by_column": missing[missing > 0].to_dict(),
            "missing_percentage": (missing / len(df) * 100).round(2).to_dict()
        }
        
        # Data type analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        profile["column_types"] = {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        }
        
        # Basic statistics for numeric columns
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe().to_dict()
            profile["numeric_stats"] = numeric_stats
        
        # Categorical column analysis
        if categorical_cols:
            cat_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                cat_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_5_values": value_counts.head(5).to_dict()
                }
            profile["categorical_stats"] = cat_stats
        
        # Potential target analysis (if target column exists)
        potential_targets = []
        for col in df.columns:
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 20:  # Potential classification target
                potential_targets.append({
                    "column": col,
                    "unique_values": unique_vals,
                    "type": "classification_candidate"
                })
            elif df[col].dtype in ['int64', 'float64'] and unique_vals > 20:
                potential_targets.append({
                    "column": col,
                    "unique_values": unique_vals,
                    "type": "regression_candidate"
                })
        
        profile["potential_targets"] = potential_targets
        
        # Save profile
        profile_path = self.handoffs_dir / "profile.json"
        hash_value = save_json(profile, profile_path)
        
        return {
            "profile_ref": str(profile_path),
            "hash": hash_value,
            "summary": {
                "rows": profile["shape"]["rows"],
                "columns": profile["shape"]["columns"],
                "missing_total": profile["missing_values"]["total_missing"],
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols)
            }
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "SchemaProfiler_profile",
                    "description": "Generate comprehensive dataset profile including shape, types, missing values, and statistics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df_ref": {
                                "type": "string",
                                "description": "Reference path to pickled DataFrame"
                            }
                        },
                        "required": ["df_ref"]
                    }
                }
            }
        ]


class QualityGuard:
    """Data quality and leakage detection."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
    
    def leakage_scan(
        self,
        df_ref: str,
        target: str,
        split_policy: str,
        time_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scan for potential data leakage issues.
        
        Args:
            df_ref: Reference to DataFrame
            target: Target column name
            split_policy: 'iid' or 'time'
            time_col: Time column for temporal splits
            
        Returns:
            Leakage scan results
        """
        df = load_pickle(df_ref)
        
        if target not in df.columns:
            return {
                "status": "block",
                "offenders": [],
                "reason": f"Target column '{target}' not found in dataset"
            }
        
        offenders = []
        warnings = []
        
        # Check for perfect correlation with target
        if df[target].dtype in ['int64', 'float64']:
            for col in df.columns:
                if col != target and df[col].dtype in ['int64', 'float64']:
                    try:
                        corr = df[col].corr(df[target])
                        if abs(corr) > 0.99:
                            offenders.append({
                                "column": col,
                                "issue": "perfect_correlation",
                                "correlation": float(corr),
                                "severity": "block"
                            })
                        elif abs(corr) > 0.95:
                            warnings.append({
                                "column": col,
                                "issue": "high_correlation",
                                "correlation": float(corr),
                                "severity": "warn"
                            })
                    except Exception:
                        pass
        
        # Check for suspicious column names
        suspicious_patterns = [
            'id', 'index', 'key', 'uuid', 'guid',
            'created', 'updated', 'modified', 'timestamp',
            'result', 'outcome', 'prediction', 'score',
            'label', 'class', 'category'
        ]
        
        for col in df.columns:
            if col != target:
                col_lower = col.lower()
                for pattern in suspicious_patterns:
                    if pattern in col_lower:
                        warnings.append({
                            "column": col,
                            "issue": "suspicious_name",
                            "pattern": pattern,
                            "severity": "warn"
                        })
                        break
        
        # Time-based leakage checks
        if split_policy == "time" and time_col:
            if time_col not in df.columns:
                offenders.append({
                    "column": time_col,
                    "issue": "missing_time_column",
                    "severity": "block"
                })
            else:
                # Check for future information
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    # Additional time-based checks could go here
                except Exception:
                    warnings.append({
                        "column": time_col,
                        "issue": "invalid_datetime_format",
                        "severity": "warn"
                    })
        
        # Determine overall status
        if offenders:
            status = "block"
        elif warnings:
            status = "warn"
        else:
            status = "pass"
        
        return {
            "status": status,
            "offenders": offenders,
            "warnings": warnings,
            "summary": {
                "blocking_issues": len(offenders),
                "warnings": len(warnings),
                "columns_checked": len(df.columns) - 1  # Exclude target
            }
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "QualityGuard_leakage_scan",
                    "description": "Scan dataset for potential data leakage issues",
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
                            "split_policy": {
                                "type": "string",
                                "enum": ["iid", "time"],
                                "description": "Data splitting policy"
                            },
                            "time_col": {
                                "type": "string",
                                "description": "Time column name for temporal splits (optional)"
                            }
                        },
                        "required": ["df_ref", "target", "split_policy"]
                    }
                }
            }
        ]
