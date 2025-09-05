"""Plotting tools for visualizations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ..utils import load_pickle


class Plotter:
    """Visualization and plotting tools."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.plots_dir = artifacts_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.rcParams['font.size'] = 10
    
    def pr_curve(
        self,
        model_ref: str,
        X_test_ref: str,
        y_test_ref: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Plot Precision-Recall curve.
        
        Args:
            model_ref: Reference to trained model
            X_test_ref: Test features reference
            y_test_ref: Test target reference
            title: Plot title
            
        Returns:
            Plot file reference and metrics
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_test = load_pickle(X_test_ref)
        y_test = load_pickle(y_test_ref)
        
        # Check if binary classification
        if len(np.unique(y_test)) != 2:
            return {
                "plot_ref": None,
                "error": "PR curve only available for binary classification"
            }
        
        # Check if model supports probabilities
        if not hasattr(model, "predict_proba"):
            return {
                "plot_ref": None,
                "error": "Model does not support probability prediction"
            }
        
        # Get probabilities
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Calculate AUC
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, y_proba)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title or 'Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.plots_dir / "pr_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_ref": str(plot_path),
            "pr_auc": float(pr_auc),
            "optimal_threshold": float(thresholds[np.argmax(precision + recall)])
        }
    
    def lift_curve(
        self,
        model_ref: str,
        X_test_ref: str,
        y_test_ref: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Plot lift curve.
        
        Args:
            model_ref: Reference to trained model
            X_test_ref: Test features reference
            y_test_ref: Test target reference
            title: Plot title
            
        Returns:
            Plot file reference and lift metrics
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_test = load_pickle(X_test_ref)
        y_test = load_pickle(y_test_ref)
        
        # Check if binary classification
        if len(np.unique(y_test)) != 2:
            return {
                "plot_ref": None,
                "error": "Lift curve only available for binary classification"
            }
        
        # Check if model supports probabilities
        if not hasattr(model, "predict_proba"):
            return {
                "plot_ref": None,
                "error": "Model does not support probability prediction"
            }
        
        # Get probabilities and sort by descending probability
        y_proba = model.predict_proba(X_test)[:, 1]
        sorted_indices = np.argsort(y_proba)[::-1]
        y_test_sorted = y_test.iloc[sorted_indices] if hasattr(y_test, 'iloc') else y_test[sorted_indices]
        
        # Calculate lift
        n_samples = len(y_test)
        n_positives = y_test.sum()
        baseline_rate = n_positives / n_samples
        
        percentiles = np.arange(0.1, 1.1, 0.1)
        lift_values = []
        
        for pct in percentiles:
            n_top = int(pct * n_samples)
            if n_top == 0:
                continue
            
            top_positives = y_test_sorted[:n_top].sum()
            top_rate = top_positives / n_top
            lift = top_rate / baseline_rate if baseline_rate > 0 else 1
            lift_values.append(lift)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(percentiles[:len(lift_values)], lift_values, 'b-', linewidth=2, marker='o')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Baseline')
        plt.xlabel('Population Percentile')
        plt.ylabel('Lift')
        plt.title(title or 'Lift Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.plots_dir / "lift_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_ref": str(plot_path),
            "max_lift": float(max(lift_values)) if lift_values else 1.0,
            "lift_at_10pct": float(lift_values[0]) if lift_values else 1.0
        }
    
    def calibration_plot(
        self,
        model_ref: str,
        X_test_ref: str,
        y_test_ref: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Plot calibration plot.
        
        Args:
            model_ref: Reference to trained model
            X_test_ref: Test features reference
            y_test_ref: Test target reference
            title: Plot title
            
        Returns:
            Plot file reference and calibration metrics
        """
        # Load model and data
        model = load_pickle(model_ref)
        X_test = load_pickle(X_test_ref)
        y_test = load_pickle(y_test_ref)
        
        # Check if binary classification
        if len(np.unique(y_test)) != 2:
            return {
                "plot_ref": None,
                "error": "Calibration plot only available for binary classification"
            }
        
        # Check if model supports probabilities
        if not hasattr(model, "predict_proba"):
            return {
                "plot_ref": None,
                "error": "Model does not support probability prediction"
            }
        
        # Get probabilities
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate calibration curve
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
            if bin_upper == 1.0:  # Include the last bin edge
                in_bin = (y_proba >= bin_lower) & (y_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(y_test[in_bin].mean())
                bin_counts.append(in_bin.sum())
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
        
        if bin_centers:
            plt.plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=8, label='Model')
            
            # Add bin counts as text
            for x, y, count in zip(bin_centers, bin_accuracies, bin_counts):
                plt.text(x, y + 0.02, str(count), ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(title or 'Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.plots_dir / "calibration_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate Expected Calibration Error
        ece = 0
        total_samples = len(y_test)
        for center, accuracy, count in zip(bin_centers, bin_accuracies, bin_counts):
            ece += abs(center - accuracy) * (count / total_samples)
        
        return {
            "plot_ref": str(plot_path),
            "ece": float(ece),
            "n_bins": len(bin_centers)
        }
    
    def bars(
        self,
        data: Dict[str, float],
        title: str,
        xlabel: str = "Categories",
        ylabel: str = "Values"
    ) -> Dict[str, Any]:
        """
        Create bar plot.
        
        Args:
            data: Dictionary of category -> value
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            Plot file reference
        """
        if not data:
            return {
                "plot_ref": None,
                "error": "No data provided for bar plot"
            }
        
        # Create plot
        plt.figure(figsize=(10, 6))
        categories = list(data.keys())
        values = list(data.values())
        
        bars = plt.bar(categories, values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_name = title.lower().replace(' ', '_').replace('/', '_') + ".png"
        plot_path = self.plots_dir / plot_name
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            "plot_ref": str(plot_path),
            "n_categories": len(categories)
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "Plotter_pr_curve",
                    "description": "Plot Precision-Recall curve for binary classification",
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
                            "title": {
                                "type": "string",
                                "description": "Plot title (optional)"
                            }
                        },
                        "required": ["model_ref", "X_test_ref", "y_test_ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Plotter_lift_curve",
                    "description": "Plot lift curve for binary classification",
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
                            "title": {
                                "type": "string",
                                "description": "Plot title (optional)"
                            }
                        },
                        "required": ["model_ref", "X_test_ref", "y_test_ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Plotter_calibration_plot",
                    "description": "Plot calibration plot for binary classification",
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
                            "title": {
                                "type": "string",
                                "description": "Plot title (optional)"
                            }
                        },
                        "required": ["model_ref", "X_test_ref", "y_test_ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Plotter_bars",
                    "description": "Create bar plot from data dictionary",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Dictionary mapping categories to values"
                            },
                            "title": {
                                "type": "string",
                                "description": "Plot title"
                            },
                            "xlabel": {
                                "type": "string",
                                "description": "X-axis label (default: Categories)"
                            },
                            "ylabel": {
                                "type": "string",
                                "description": "Y-axis label (default: Values)"
                            }
                        },
                        "required": ["data", "title"]
                    }
                }
            }
        ]
