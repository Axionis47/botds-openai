"""Configuration management with Pydantic validation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Data source configuration."""
    source: Literal["builtin", "csv"] = "builtin"
    name: str = "iris"  # iris | breast_cancer | diabetes | csv
    csv_paths: List[str] = Field(default_factory=list)
    target: str = ""  # auto for builtins; required for csv


class BudgetConfig(BaseModel):
    """Resource budget limits."""
    time_min: int = 25
    memory_gb: float = 4.0
    token_budget: int = 8000


class FairnessConfig(BaseModel):
    """Fairness evaluation settings."""
    enabled: bool = False
    sensitive_cols: List[str] = Field(default_factory=list)
    policy: Literal["report", "block"] = "report"


class PIIConfig(BaseModel):
    """PII detection and handling."""
    enabled: bool = True
    patterns: List[str] = Field(default_factory=lambda: ["email", "phone"])
    action: Literal["redact", "block"] = "redact"


class SplitConfig(BaseModel):
    """Data splitting configuration."""
    policy: Literal["iid", "time"] = "iid"
    time_col: str = ""
    test_size: float = 0.2
    val_size: float = 0.2
    rolling: bool = False
    seed: int = 42


class SamplingConfig(BaseModel):
    """Sampling configuration for large datasets."""
    eda_rows: int = 200000
    stratify_by: List[str] = Field(default_factory=list)


class CacheConfig(BaseModel):
    """Cache behavior settings."""
    mode: Literal["warm", "cold", "paranoid"] = "warm"
    dir: str = "./cache"


class ReportConfig(BaseModel):
    """Report generation settings."""
    out_dir: str = "./artifacts"
    format: Literal["html", "md"] = "html"


class LLMConfig(BaseModel):
    """LLM configuration."""
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"


class Config(BaseModel):
    """Main configuration model."""
    data: DataConfig = Field(default_factory=DataConfig)
    task: Literal["auto", "classification", "regression"] = "auto"
    metrics: Dict[str, Any] = Field(default_factory=lambda: {
        "primary": "auto",
        "secondary": ["roc_auc", "accuracy", "rmse"]
    })
    business_goal: str = "Plain English goal"
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    fairness: FairnessConfig = Field(default_factory=FairnessConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    llms: LLMConfig = Field(default_factory=LLMConfig)

    @validator("data")
    def validate_data_config(cls, v: DataConfig) -> DataConfig:
        """Validate data configuration."""
        if v.source == "csv" and not v.csv_paths:
            raise ValueError("csv_paths required when source=csv")
        if v.source == "csv" and not v.target:
            raise ValueError("target required when source=csv")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)

    def validate_environment(self) -> None:
        """Validate required environment variables."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "OpenAI is the sole decision authority - no fallback available."
            )
