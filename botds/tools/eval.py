"""Advanced evaluation tools: calibration, fairness, robustness."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from ..utils import load_pickle, save_pickle


class Calibrator:
    """Model calibration tools."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.models_dir = artifacts_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(
        self,
        model_ref: str,
        X_val_ref: str,
        y_val_ref: str,
        method: str = "isotonic"
    ) -> Dict[str, Any]:
        """
        Fit calibration on validation set.
        
        Args:
            model_ref: Reference to trained model
            X_val_ref: Validation features reference
            y_val_ref: Validation target reference
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Calibrated model reference and calibration metrics
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_val = load_pickle(X_val_ref)
        y_val = load_pickle(y_val_ref)
        
        # Check if model supports probability prediction
        if not hasattr(model, "predict_proba"):
            return {
                "calibrated_model_ref": model_ref,  # Return original
                "calibration_applied": False,
                "reason": "Model does not support probability prediction"
            }
        
        # Fit calibration
        calibrated_model = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrated_model.fit(X_val, y_val)
        
        # Save calibrated model
        cal_model_id = f"calibrated_{hash(model_ref)}_{method}"[:16]
        cal_model_path = self.models_dir / f"{cal_model_id}.pkl"
        save_pickle(calibrated_model, cal_model_path)
        
        # Calculate calibration metrics
        y_proba_orig = model.predict_proba(X_val)[:, 1]
        y_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
        
        # Expected Calibration Error (ECE)
        ece_orig = self._calculate_ece(y_val, y_proba_orig)
        ece_cal = self._calculate_ece(y_val, y_proba_cal)
        
        # Brier Score
        brier_orig = brier_score_loss(y_val, y_proba_orig)
        brier_cal = brier_score_loss(y_val, y_proba_cal)
        
        return {
            "calibrated_model_ref": str(cal_model_path),
            "calibration_applied": True,
            "method": method,
            "metrics": {
                "ece_before": float(ece_orig),
                "ece_after": float(ece_cal),
                "ece_improvement": float(ece_orig - ece_cal),
                "brier_before": float(brier_orig),
                "brier_after": float(brier_cal),
                "brier_improvement": float(brier_orig - brier_cal)
            }
        }
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Calibrator_fit",
                    "description": "Fit model calibration and calculate calibration metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_ref": {
                                "type": "string",
                                "description": "Reference to trained model"
                            },
                            "X_val_ref": {
                                "type": "string",
                                "description": "Reference to validation features"
                            },
                            "y_val_ref": {
                                "type": "string",
                                "description": "Reference to validation target"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["isotonic", "sigmoid"],
                                "description": "Calibration method (default: isotonic)"
                            }
                        },
                        "required": ["model_ref", "X_val_ref", "y_val_ref"]
                    }
                }
            }
        ]


class Fairness:
    """Fairness evaluation tools."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
    
    def slice_metrics(
        self,
        model_ref: str,
        X_test_ref: str,
        y_test_ref: str,
        sensitive_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate metrics across sensitive attribute slices.
        
        Args:
            model_ref: Reference to trained model
            X_test_ref: Test features reference
            y_test_ref: Test target reference
            sensitive_cols: List of sensitive column names
            
        Returns:
            Per-slice metrics analysis
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_test = load_pickle(X_test_ref)
        y_test = load_pickle(y_test_ref)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Check which sensitive columns exist
        available_cols = [col for col in sensitive_cols if col in X_test.columns]
        
        if not available_cols:
            return {
                "slice_metrics": {},
                "fairness_summary": {
                    "status": "no_sensitive_columns",
                    "message": "No sensitive columns found in test data"
                }
            }
        
        slice_results = {}
        
        for col in available_cols:
            col_results = {}
            unique_values = X_test[col].unique()
            
            for value in unique_values:
                mask = X_test[col] == value
                if mask.sum() == 0:
                    continue
                
                y_true_slice = y_test[mask]
                y_pred_slice = y_pred[mask]
                
                # Calculate metrics for this slice
                if len(np.unique(y_true_slice)) <= 20:  # Classification
                    from sklearn.metrics import accuracy_score, f1_score
                    metrics = {
                        "accuracy": float(accuracy_score(y_true_slice, y_pred_slice)),
                        "f1_score": float(f1_score(y_true_slice, y_pred_slice, average='weighted')),
                        "sample_size": int(mask.sum())
                    }
                else:  # Regression
                    from sklearn.metrics import mean_absolute_error
                    metrics = {
                        "mae": float(mean_absolute_error(y_true_slice, y_pred_slice)),
                        "sample_size": int(mask.sum())
                    }
                
                col_results[str(value)] = metrics
            
            slice_results[col] = col_results
        
        # Calculate fairness summary
        fairness_summary = self._analyze_fairness(slice_results)
        
        return {
            "slice_metrics": slice_results,
            "fairness_summary": fairness_summary
        }
    
    def _analyze_fairness(self, slice_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fairness across slices."""
        issues = []
        
        for col, col_results in slice_results.items():
            if len(col_results) < 2:
                continue
            
            # Get primary metric values
            metric_values = []
            for slice_name, metrics in col_results.items():
                if "accuracy" in metrics:
                    metric_values.append(("accuracy", metrics["accuracy"]))
                elif "mae" in metrics:
                    metric_values.append(("mae", metrics["mae"]))
            
            if len(metric_values) >= 2:
                metric_name = metric_values[0][0]
                values = [v[1] for v in metric_values]
                
                # Calculate disparity
                max_val = max(values)
                min_val = min(values)
                
                if metric_name == "accuracy":
                    disparity = max_val - min_val
                    if disparity > 0.1:  # 10% accuracy gap
                        issues.append({
                            "column": col,
                            "metric": metric_name,
                            "disparity": float(disparity),
                            "severity": "high" if disparity > 0.2 else "medium"
                        })
                elif metric_name == "mae":
                    disparity = max_val / min_val if min_val > 0 else float('inf')
                    if disparity > 1.5:  # 50% relative difference
                        issues.append({
                            "column": col,
                            "metric": metric_name,
                            "disparity_ratio": float(disparity),
                            "severity": "high" if disparity > 2.0 else "medium"
                        })
        
        return {
            "status": "issues_found" if issues else "no_issues",
            "issues": issues,
            "total_issues": len(issues)
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Fairness_slice_metrics",
                    "description": "Calculate metrics across sensitive attribute slices for fairness analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_ref": {
                                "type": "string",
                                "description": "Reference to trained model"
                            },
                            "X_test_ref": {
                                "type": "string",
                                "description": "Reference to test features"
                            },
                            "y_test_ref": {
                                "type": "string",
                                "description": "Reference to test target"
                            },
                            "sensitive_cols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of sensitive column names"
                            }
                        },
                        "required": ["model_ref", "X_test_ref", "y_test_ref", "sensitive_cols"]
                    }
                }
            }
        ]


class Robustness:
    """Model robustness evaluation."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
    
    def ablation(
        self,
        model_ref: str,
        X_val_ref: str,
        y_val_ref: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Feature ablation study.
        
        Args:
            model_ref: Reference to trained model
            X_val_ref: Validation features reference
            y_val_ref: Validation target reference
            top_k: Number of top features to ablate
            
        Returns:
            Feature importance and ablation results
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_val = load_pickle(X_val_ref)
        y_val = load_pickle(y_val_ref)
        
        # Get baseline performance
        y_pred_baseline = model.predict(X_val)
        
        # Calculate baseline metric
        if len(np.unique(y_val)) <= 20:  # Classification
            from sklearn.metrics import accuracy_score
            baseline_score = accuracy_score(y_val, y_pred_baseline)
            metric_name = "accuracy"
        else:  # Regression
            from sklearn.metrics import mean_absolute_error
            baseline_score = mean_absolute_error(y_val, y_pred_baseline)
            metric_name = "mae"
        
        # Feature importance (if available)
        feature_importance = {}
        if hasattr(model, "feature_importances_"):
            importance_scores = model.feature_importances_
            feature_names = X_val.columns if hasattr(X_val, 'columns') else [f"feature_{i}" for i in range(X_val.shape[1])]
            feature_importance = dict(zip(feature_names, importance_scores))
        
        # Ablation study
        ablation_results = []
        
        # Get top features to ablate
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            # Use all features if no importance available
            feature_names = X_val.columns if hasattr(X_val, 'columns') else [f"feature_{i}" for i in range(X_val.shape[1])]
            top_features = [(name, 1.0) for name in feature_names[:top_k]]
        
        for feature_name, importance in top_features:
            # Create ablated dataset (set feature to mean/mode)
            X_ablated = X_val.copy()
            
            if hasattr(X_val, 'columns'):
                col_idx = list(X_val.columns).index(feature_name)
            else:
                col_idx = int(feature_name.split('_')[1]) if 'feature_' in feature_name else 0
            
            # Replace with mean for numeric, mode for categorical
            if hasattr(X_ablated, 'iloc'):
                col_data = X_ablated.iloc[:, col_idx]
                if pd.api.types.is_numeric_dtype(col_data):
                    replacement_value = col_data.mean()
                else:
                    replacement_value = col_data.mode().iloc[0] if len(col_data.mode()) > 0 else col_data.iloc[0]
                X_ablated.iloc[:, col_idx] = replacement_value
            else:
                # NumPy array
                col_data = X_ablated[:, col_idx]
                replacement_value = np.mean(col_data)
                X_ablated[:, col_idx] = replacement_value
            
            # Get predictions with ablated feature
            y_pred_ablated = model.predict(X_ablated)
            
            # Calculate performance drop
            if metric_name == "accuracy":
                ablated_score = accuracy_score(y_val, y_pred_ablated)
                performance_drop = baseline_score - ablated_score
            else:  # MAE
                ablated_score = mean_absolute_error(y_val, y_pred_ablated)
                performance_drop = ablated_score - baseline_score  # Higher MAE is worse
            
            ablation_results.append({
                "feature": feature_name,
                "importance": float(importance),
                "baseline_score": float(baseline_score),
                "ablated_score": float(ablated_score),
                "performance_drop": float(performance_drop)
            })
        
        return {
            "baseline_score": float(baseline_score),
            "metric_name": metric_name,
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "ablation_results": ablation_results,
            "summary": {
                "max_performance_drop": float(max([r["performance_drop"] for r in ablation_results], default=0)),
                "avg_performance_drop": float(np.mean([r["performance_drop"] for r in ablation_results])) if ablation_results else 0
            }
        }
    
    def shock_tests(
        self,
        model_ref: str,
        X_val_ref: str,
        y_val_ref: str
    ) -> Dict[str, Any]:
        """
        Robustness shock tests.
        
        Args:
            model_ref: Reference to trained model
            X_val_ref: Validation features reference
            y_val_ref: Validation target reference
            
        Returns:
            Robustness grade and test results
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_val = load_pickle(X_val_ref)
        y_val = load_pickle(y_val_ref)
        
        # Baseline performance
        y_pred_baseline = model.predict(X_val)
        
        if len(np.unique(y_val)) <= 20:  # Classification
            from sklearn.metrics import accuracy_score
            baseline_score = accuracy_score(y_val, y_pred_baseline)
        else:  # Regression
            from sklearn.metrics import mean_absolute_error
            baseline_score = mean_absolute_error(y_val, y_pred_baseline)
        
        shock_results = []
        
        # Test 1: Missing values shock
        X_missing = X_val.copy()
        if hasattr(X_missing, 'iloc'):
            # Randomly set 10% of values to NaN
            mask = np.random.random(X_missing.shape) < 0.1
            X_missing = X_missing.mask(mask)
            X_missing = X_missing.fillna(X_missing.mean())  # Simple imputation
        
        try:
            y_pred_missing = model.predict(X_missing)
            if len(np.unique(y_val)) <= 20:
                missing_score = accuracy_score(y_val, y_pred_missing)
                missing_drop = baseline_score - missing_score
            else:
                missing_score = mean_absolute_error(y_val, y_pred_missing)
                missing_drop = missing_score - baseline_score
            
            shock_results.append({
                "test": "missing_values",
                "performance_drop": float(missing_drop),
                "passed": missing_drop < 0.1
            })
        except Exception:
            shock_results.append({
                "test": "missing_values",
                "performance_drop": float('inf'),
                "passed": False
            })
        
        # Test 2: Noise injection
        if hasattr(X_val, 'select_dtypes'):
            numeric_cols = X_val.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_noisy = X_val.copy()
                for col in numeric_cols:
                    noise = np.random.normal(0, X_val[col].std() * 0.1, len(X_val))
                    X_noisy[col] = X_val[col] + noise
                
                try:
                    y_pred_noisy = model.predict(X_noisy)
                    if len(np.unique(y_val)) <= 20:
                        noisy_score = accuracy_score(y_val, y_pred_noisy)
                        noisy_drop = baseline_score - noisy_score
                    else:
                        noisy_score = mean_absolute_error(y_val, y_pred_noisy)
                        noisy_drop = noisy_score - baseline_score
                    
                    shock_results.append({
                        "test": "noise_injection",
                        "performance_drop": float(noisy_drop),
                        "passed": noisy_drop < 0.05
                    })
                except Exception:
                    shock_results.append({
                        "test": "noise_injection",
                        "performance_drop": float('inf'),
                        "passed": False
                    })
        
        # Calculate robustness grade
        passed_tests = sum(1 for test in shock_results if test["passed"])
        total_tests = len(shock_results)
        
        if total_tests == 0:
            grade = "C"
        elif passed_tests == total_tests:
            grade = "A"
        elif passed_tests >= total_tests * 0.75:
            grade = "B"
        elif passed_tests >= total_tests * 0.5:
            grade = "C"
        else:
            grade = "D"
        
        return {
            "resilience_grade": grade,
            "baseline_score": float(baseline_score),
            "shock_results": shock_results,
            "summary": {
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "pass_rate": float(passed_tests / total_tests) if total_tests > 0 else 0
            }
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Robustness_ablation",
                    "description": "Perform feature ablation study to assess feature importance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_ref": {
                                "type": "string",
                                "description": "Reference to trained model"
                            },
                            "X_val_ref": {
                                "type": "string",
                                "description": "Reference to validation features"
                            },
                            "y_val_ref": {
                                "type": "string",
                                "description": "Reference to validation target"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top features to ablate (default: 5)"
                            }
                        },
                        "required": ["model_ref", "X_val_ref", "y_val_ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Robustness_shock_tests",
                    "description": "Run robustness shock tests and assign resilience grade",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_ref": {
                                "type": "string",
                                "description": "Reference to trained model"
                            },
                            "X_val_ref": {
                                "type": "string",
                                "description": "Reference to validation features"
                            },
                            "y_val_ref": {
                                "type": "string",
                                "description": "Reference to validation target"
                            }
                        },
                        "required": ["model_ref", "X_val_ref", "y_val_ref"]
                    }
                }
            }
        ]
