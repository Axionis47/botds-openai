"""Project context management: charter, cards, decision log, manifest."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import get_timestamp, save_json


class DecisionLog:
    """Log for critical decisions made by OpenAI."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def record_decision(
        self,
        stage: str,
        decision: str,
        rationale: str,
        inputs_refs: List[str],
        auth_model: str = "openai"
    ) -> None:
        """Record a critical decision."""
        entry = {
            "stage": stage,
            "decision": decision,
            "rationale": rationale,
            "inputs_refs": inputs_refs,
            "timestamp": get_timestamp(),
            "auth_model": auth_model
        }
        
        # Append to JSONL file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_decisions(self) -> List[Dict[str, Any]]:
        """Get all recorded decisions."""
        if not self.log_path.exists():
            return []
        
        decisions = []
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        return decisions


class DataCard:
    """Data card for dataset documentation."""
    
    def __init__(self, name: str, version: str = "v1"):
        self.name = name
        self.version = version
        self.created_at = get_timestamp()
        self.metadata: Dict[str, Any] = {}
        self.quality_notes: List[str] = []
        self.leakage_rules: List[str] = []
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry."""
        self.metadata[key] = value
    
    def add_quality_note(self, note: str) -> None:
        """Add quality observation."""
        self.quality_notes.append(note)
    
    def add_leakage_rule(self, rule: str) -> None:
        """Add leakage prevention rule."""
        self.leakage_rules.append(rule)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "quality_notes": self.quality_notes,
            "leakage_rules": self.leakage_rules
        }
    
    def save(self, path: Path) -> str:
        """Save data card and return hash."""
        return save_json(self.to_dict(), path)


class HandoffLedger:
    """Ledger for tracking handoffs between pipeline stages."""
    
    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
    
    def append(
        self,
        job_id: str,
        stage: str,
        input_refs: List[str],
        output_refs: List[str],
        schema_uri: str,
        hash_value: str
    ) -> None:
        """Append a handoff entry."""
        entry = {
            "job_id": job_id,
            "stage": stage,
            "inputs": input_refs,
            "outputs": output_refs,
            "schema": schema_uri,
            "hash": hash_value,
            "timestamp": get_timestamp()
        }
        
        self.entries.append(entry)
        
        # Append to JSONL file
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all ledger entries."""
        if not self.ledger_path.exists():
            return []
        
        entries = []
        with open(self.ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries


class RunManifest:
    """Manifest for a complete pipeline run."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.created_at = get_timestamp()
        self.config_hash: Optional[str] = None
        self.dataset_hash: Optional[str] = None
        self.seeds: Dict[str, int] = {}
        self.budgets_used: Dict[str, Any] = {}
        self.cache_hits: Dict[str, bool] = {}
        self.versions: Dict[str, str] = {}
        self.shortcuts_taken: List[str] = []
    
    def set_config_hash(self, hash_value: str) -> None:
        """Set configuration hash."""
        self.config_hash = hash_value
    
    def set_dataset_hash(self, hash_value: str) -> None:
        """Set dataset hash."""
        self.dataset_hash = hash_value
    
    def add_seed(self, component: str, seed: int) -> None:
        """Add seed used by a component."""
        self.seeds[component] = seed
    
    def add_budget_usage(self, stage: str, usage: Dict[str, Any]) -> None:
        """Add budget usage for a stage."""
        self.budgets_used[stage] = usage
    
    def add_cache_hit(self, stage: str, hit: bool) -> None:
        """Record cache hit/miss for a stage."""
        self.cache_hits[stage] = hit
    
    def add_version(self, component: str, version: str) -> None:
        """Add version information."""
        self.versions[component] = version
    
    def add_shortcut(self, description: str) -> None:
        """Add shortcut taken due to budget constraints."""
        self.shortcuts_taken.append(description)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "seeds": self.seeds,
            "budgets_used": self.budgets_used,
            "cache_hits": self.cache_hits,
            "versions": self.versions,
            "shortcuts_taken": self.shortcuts_taken
        }
    
    def save(self, path: Path) -> str:
        """Save manifest and return hash."""
        return save_json(self.to_dict(), path)
