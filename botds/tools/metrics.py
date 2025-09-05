"""Metrics calculation and confidence interval tools."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from ..utils import load_pickle


class Metrics:
    """Metrics calculation and statistical analysis."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
    
    def evaluate(
        self,
        model_ref: str,
        X_test_ref: str,
        y_test_ref: str,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set with comprehensive metrics.
        
        Args:
            model_ref: Reference to trained model
            X_test_ref: Test features reference
            y_test_ref: Test target reference
            task_type: 'classification' or 'regression' (auto-inferred if None)
            
        Returns:
            Comprehensive evaluation metrics
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_test = load_pickle(X_test_ref)
        y_test = load_pickle(y_test_ref)
        
        # Infer task type if not provided
        if task_type is None:
            task_type = "classification" if len(np.unique(y_test)) <= 20 else "regression"
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        metrics = {"task_type": task_type}
        
        if task_type == "classification":
            # Classification metrics
            metrics.update({
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average='weighted')),
                "precision": float(precision_score(y_test, y_pred, average='weighted')),
                "recall": float(recall_score(y_test, y_pred, average='weighted'))
            })
            
            # ROC AUC and PR AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
                        metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
                except Exception:
                    pass
        
        else:
            # Regression metrics
            metrics.update({
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": float(r2_score(y_test, y_pred))
            })
            
            # Additional regression metrics
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
            metrics["mape"] = float(mape)
        
        return metrics
    
    def bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for a metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric_name: Name of metric to calculate
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Mean, lower, and upper bounds
        """
        metric_func = self._get_metric_function(metric_name)
        
        # Bootstrap sampling
        n_samples = len(y_true)
        bootstrap_scores = []
        
        np.random.seed(42)  # For reproducibility
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except Exception:
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            "mean": float(np.mean(bootstrap_scores)),
            "lower": float(np.percentile(bootstrap_scores, lower_percentile)),
            "upper": float(np.percentile(bootstrap_scores, upper_percentile)),
            "std": float(np.std(bootstrap_scores)),
            "n_bootstrap": len(bootstrap_scores)
        }
    
    def _get_metric_function(self, metric_name: str):
        """Get metric function by name."""
        metric_functions = {
            "accuracy": accuracy_score,
            "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            "mae": mean_absolute_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score
        }
        
        if metric_name not in metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        return metric_functions[metric_name]
    
    def compare_models(
        self,
        model_results: List[Dict[str, Any]],
        primary_metric: str
    ) -> Dict[str, Any]:
        """
        Compare multiple models and rank them.
        
        Args:
            model_results: List of model evaluation results
            primary_metric: Primary metric for ranking
            
        Returns:
            Ranked comparison results
        """
        if not model_results:
            return {"rankings": [], "best_model": None}
        
        # Extract primary metric scores
        scores = []
        for result in model_results:
            metrics = result.get("val_metrics", {})
            score = metrics.get(primary_metric)
            if score is not None:
                scores.append((score, result))
        
        # Sort by primary metric (higher is better for most metrics except MAE, RMSE)
        reverse_sort = primary_metric not in ["mae", "rmse", "mape"]
        scores.sort(key=lambda x: x[0], reverse=reverse_sort)
        
        # Create rankings
        rankings = []
        for rank, (score, result) in enumerate(scores, 1):
            rankings.append({
                "rank": rank,
                "model_id": result.get("model_id", f"model_{rank}"),
                "model_spec": result.get("model_spec", {}),
                "primary_score": score,
                "all_metrics": result.get("val_metrics", {}),
                "train_time": result.get("train_time_seconds", 0)
            })
        
        best_model = rankings[0] if rankings else None
        
        return {
            "rankings": rankings,
            "best_model": best_model,
            "primary_metric": primary_metric,
            "total_models": len(rankings)
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Metrics_evaluate",
                    "description": "Evaluate model on test set with comprehensive metrics",
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
                            "task_type": {
                                "type": "string",
                                "enum": ["classification", "regression"],
                                "description": "Task type (auto-inferred if not provided)"
                            }
                        },
                        "required": ["model_ref", "X_test_ref", "y_test_ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Metrics_bootstrap_ci",
                    "description": "Calculate bootstrap confidence intervals for a metric",
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
                            "metric_name": {
                                "type": "string",
                                "enum": ["accuracy", "f1_score", "precision", "recall", "mae", "rmse", "r2"],
                                "description": "Metric to calculate CI for"
                            },
                            "n_bootstrap": {
                                "type": "integer",
                                "description": "Number of bootstrap samples (default: 1000)"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence level (default: 0.95)"
                            }
                        },
                        "required": ["model_ref", "X_test_ref", "y_test_ref", "metric_name"]
                    }
                }
            }
        ]
