"""Budget monitoring and control tools."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import get_memory_usage_gb, get_timestamp


class BudgetGuard:
    """Budget monitoring and enforcement."""
    
    def __init__(self, artifacts_dir: Path, budgets: Dict[str, Any]):
        self.artifacts_dir = artifacts_dir
        self.budgets = budgets
        self.start_time = time.time()
        self.checkpoints: List[Dict[str, Any]] = []
        self.token_usage = 0
        
        # Budget limits
        self.time_limit_seconds = budgets.get("time_min", 25) * 60
        self.memory_limit_gb = budgets.get("memory_gb", 4.0)
        self.token_limit = budgets.get("token_budget", 8000)
    
    def checkpoint(
        self,
        stage: str,
        additional_tokens: int = 0
    ) -> Dict[str, Any]:
        """
        Check budget status at a pipeline stage.
        
        Args:
            stage: Current pipeline stage
            additional_tokens: Additional tokens used in this stage
            
        Returns:
            Budget status and recommendations
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        current_memory = get_memory_usage_gb()
        self.token_usage += additional_tokens
        
        # Create checkpoint record
        checkpoint = {
            "stage": stage,
            "timestamp": get_timestamp(),
            "elapsed_seconds": elapsed_time,
            "memory_gb": current_memory,
            "tokens_used": self.token_usage,
            "budgets": {
                "time_remaining_seconds": max(0, self.time_limit_seconds - elapsed_time),
                "memory_remaining_gb": max(0, self.memory_limit_gb - current_memory),
                "tokens_remaining": max(0, self.token_limit - self.token_usage)
            }
        }
        
        self.checkpoints.append(checkpoint)
        
        # Determine status and recommendations
        status = "ok"
        recommendations = []
        
        # Time budget check
        time_usage_pct = elapsed_time / self.time_limit_seconds
        if time_usage_pct > 0.9:
            status = "abort"
            recommendations.append("Time budget exceeded - consider aborting")
        elif time_usage_pct > 0.7:
            status = "downshift"
            recommendations.append("Time budget tight - consider reducing model complexity")
        
        # Memory budget check
        memory_usage_pct = current_memory / self.memory_limit_gb
        if memory_usage_pct > 0.9:
            status = "abort"
            recommendations.append("Memory budget exceeded - consider smaller dataset")
        elif memory_usage_pct > 0.7:
            status = "downshift"
            recommendations.append("Memory usage high - consider feature reduction")
        
        # Token budget check
        token_usage_pct = self.token_usage / self.token_limit
        if token_usage_pct > 0.9:
            status = "abort"
            recommendations.append("Token budget exceeded - reduce LLM calls")
        elif token_usage_pct > 0.7:
            status = "downshift"
            recommendations.append("Token usage high - simplify remaining decisions")
        
        # Overall status (most restrictive wins)
        if any("abort" in rec for rec in recommendations):
            status = "abort"
        elif any("downshift" in rec for rec in recommendations):
            status = "downshift"
        
        return {
            "status": status,
            "stage": stage,
            "usage": {
                "time_pct": round(time_usage_pct * 100, 1),
                "memory_pct": round(memory_usage_pct * 100, 1),
                "tokens_pct": round(token_usage_pct * 100, 1)
            },
            "recommendations": recommendations,
            "checkpoint": checkpoint
        }
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get overall budget usage summary."""
        if not self.checkpoints:
            return {"status": "no_checkpoints"}
        
        latest = self.checkpoints[-1]
        
        return {
            "total_stages": len(self.checkpoints),
            "final_usage": {
                "time_seconds": latest["elapsed_seconds"],
                "time_pct": round(latest["elapsed_seconds"] / self.time_limit_seconds * 100, 1),
                "memory_gb": latest["memory_gb"],
                "memory_pct": round(latest["memory_gb"] / self.memory_limit_gb * 100, 1),
                "tokens": latest["tokens_used"],
                "tokens_pct": round(latest["tokens_used"] / self.token_limit * 100, 1)
            },
            "checkpoints": self.checkpoints
        }
    
    def suggest_shortcuts(self, remaining_stages: List[str]) -> List[str]:
        """Suggest shortcuts for remaining stages based on budget constraints."""
        shortcuts = []
        
        latest_checkpoint = self.checkpoints[-1] if self.checkpoints else None
        if not latest_checkpoint:
            return shortcuts
        
        usage = latest_checkpoint["budgets"]
        
        # Time-based shortcuts
        if usage["time_remaining_seconds"] < 300:  # Less than 5 minutes
            shortcuts.extend([
                "Skip hyperparameter tuning - use default parameters",
                "Reduce cross-validation folds from 5 to 3",
                "Skip robustness testing - use basic evaluation only"
            ])
        
        # Memory-based shortcuts
        if usage["memory_remaining_gb"] < 1.0:  # Less than 1GB
            shortcuts.extend([
                "Sample dataset to 50% for remaining stages",
                "Use simpler models (linear instead of tree-based)",
                "Skip feature importance analysis"
            ])
        
        # Token-based shortcuts
        if usage["tokens_remaining"] < 1000:  # Less than 1000 tokens
            shortcuts.extend([
                "Use cached decisions where possible",
                "Simplify report generation - basic template only",
                "Skip detailed explanations in decision log"
            ])
        
        return shortcuts
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "BudgetGuard_checkpoint",
                    "description": "Check budget status at pipeline stage and get recommendations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stage": {
                                "type": "string",
                                "description": "Current pipeline stage name"
                            },
                            "additional_tokens": {
                                "type": "integer",
                                "description": "Additional tokens used in this stage (default: 0)"
                            }
                        },
                        "required": ["stage"]
                    }
                }
            }
        ]
