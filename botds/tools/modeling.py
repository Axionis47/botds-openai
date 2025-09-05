"""Model training and hyperparameter tuning tools."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from ..utils import load_pickle, save_pickle


class ModelTrainer:
    """Model training utilities."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.models_dir = artifacts_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        model_spec: Dict[str, Any],
        X_train_ref: str,
        y_train_ref: str,
        X_val_ref: str,
        y_val_ref: str
    ) -> Dict[str, Any]:
        """
        Train a model with given specification.
        
        Args:
            model_spec: Model specification dict
            X_train_ref: Training features reference
            y_train_ref: Training target reference
            X_val_ref: Validation features reference
            y_val_ref: Validation target reference
            
        Returns:
            Training results with validation metrics
        """
        # Load data
        X_train = load_pickle(X_train_ref)
        y_train = load_pickle(y_train_ref)
        X_val = load_pickle(X_val_ref)
        y_val = load_pickle(y_val_ref)
        
        # Create model
        model = self._create_model(model_spec)
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Validate
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        task_type = self._infer_task_type(y_train)
        val_metrics = self._calculate_metrics(y_val, y_val_pred, task_type)
        
        # Save model
        model_id = f"{model_spec['name']}_{hash(str(model_spec))}"[:16]
        model_path = self.models_dir / f"{model_id}.pkl"
        save_pickle(model, model_path)
        
        return {
            "model_ref": str(model_path),
            "model_id": model_id,
            "model_spec": model_spec,
            "val_metrics": val_metrics,
            "train_time_seconds": train_time,
            "task_type": task_type
        }
    
    def _create_model(self, spec: Dict[str, Any]) -> Any:
        """Create model from specification."""
        name = spec["name"]
        params = spec.get("params", {})
        
        if name == "logistic_regression":
            return LogisticRegression(**params, random_state=42)
        elif name == "linear_regression":
            return LinearRegression(**params)
        elif name == "random_forest_classifier":
            return RandomForestClassifier(**params, random_state=42)
        elif name == "random_forest_regressor":
            return RandomForestRegressor(**params, random_state=42)
        else:
            raise ValueError(f"Unknown model: {name}")
    
    def _infer_task_type(self, y: np.ndarray) -> str:
        """Infer if task is classification or regression."""
        if len(np.unique(y)) <= 20 and y.dtype in ['int64', 'object']:
            return "classification"
        else:
            return "regression"
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate appropriate metrics."""
        metrics = {}
        
        if task_type == "classification":
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average='weighted'))
        else:
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        return metrics
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "ModelTrainer_train",
                    "description": "Train a model with given specification and return validation metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_spec": {
                                "type": "object",
                                "description": "Model specification with name and params",
                                "properties": {
                                    "name": {"type": "string"},
                                    "params": {"type": "object"}
                                },
                                "required": ["name"]
                            },
                            "X_train_ref": {
                                "type": "string",
                                "description": "Reference to training features"
                            },
                            "y_train_ref": {
                                "type": "string", 
                                "description": "Reference to training target"
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
                        "required": ["model_spec", "X_train_ref", "y_train_ref", "X_val_ref", "y_val_ref"]
                    }
                }
            }
        ]


class Tuner:
    """Hyperparameter tuning utilities."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
    
    def quick_search(
        self,
        model_family: str,
        X_train_ref: str,
        y_train_ref: str,
        budget_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Quick hyperparameter search within budget.
        
        Args:
            model_family: Model family name
            X_train_ref: Training features reference
            y_train_ref: Training target reference
            budget_minutes: Time budget in minutes
            
        Returns:
            Best model spec and trial results
        """
        X_train = load_pickle(X_train_ref)
        y_train = load_pickle(y_train_ref)
        
        # Define search spaces
        search_spaces = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            },
            "logistic_regression": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "lbfgs"]
            }
        }
        
        if model_family not in search_spaces:
            raise ValueError(f"Unknown model family: {model_family}")
        
        # Infer task type and create base model
        task_type = "classification" if len(np.unique(y_train)) <= 20 else "regression"
        
        if model_family == "random_forest":
            if task_type == "classification":
                base_model = RandomForestClassifier(random_state=42)
            else:
                base_model = RandomForestRegressor(random_state=42)
        elif model_family == "logistic_regression":
            base_model = LogisticRegression(random_state=42)
        
        # Grid search with time budget
        param_grid = search_spaces[model_family]
        
        # Adjust for regression models
        if task_type == "regression" and model_family == "logistic_regression":
            # Use linear regression instead
            from sklearn.linear_model import Ridge
            base_model = Ridge(random_state=42)
            param_grid = {"alpha": [0.1, 1.0, 10.0]}
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='accuracy' if task_type == "classification" else 'neg_mean_absolute_error',
            n_jobs=-1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        # Extract results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        trials = []
        for i, (params, score) in enumerate(zip(grid_search.cv_results_['params'], 
                                               grid_search.cv_results_['mean_test_score'])):
            trials.append({
                "trial_id": i,
                "params": params,
                "score": float(score),
                "rank": int(grid_search.cv_results_['rank_test_score'][i])
            })
        
        # Create best model spec
        model_name = f"{model_family}_{'classifier' if task_type == 'classification' else 'regressor'}"
        if model_family == "logistic_regression" and task_type == "regression":
            model_name = "ridge_regression"
        
        best_spec = {
            "name": model_name,
            "params": best_params
        }
        
        return {
            "best_spec": best_spec,
            "best_score": float(best_score),
            "search_time_seconds": search_time,
            "trials": trials,
            "total_trials": len(trials)
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Tuner_quick_search",
                    "description": "Perform quick hyperparameter search within time budget",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_family": {
                                "type": "string",
                                "enum": ["random_forest", "logistic_regression"],
                                "description": "Model family to tune"
                            },
                            "X_train_ref": {
                                "type": "string",
                                "description": "Reference to training features"
                            },
                            "y_train_ref": {
                                "type": "string",
                                "description": "Reference to training target"
                            },
                            "budget_minutes": {
                                "type": "integer",
                                "description": "Time budget in minutes (default: 5)"
                            }
                        },
                        "required": ["model_family", "X_train_ref", "y_train_ref"]
                    }
                }
            }
        ]
