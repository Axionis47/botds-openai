"""PII detection and redaction tools."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..utils import load_pickle, save_pickle


class PII:
    """PII detection and handling tools."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.data_dir = artifacts_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # PII patterns
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "url": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'
        }
    
    def scan(
        self,
        df_ref: str,
        patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Scan DataFrame for PII patterns.
        
        Args:
            df_ref: Reference to DataFrame
            patterns: List of pattern names to check (default: all)
            
        Returns:
            PII scan results
        """
        df = load_pickle(df_ref)
        patterns = patterns or list(self.patterns.keys())
        
        findings = {}
        total_matches = 0
        
        for pattern_name in patterns:
            if pattern_name not in self.patterns:
                continue
            
            pattern = self.patterns[pattern_name]
            pattern_findings = {}
            
            # Check each column
            for col in df.columns:
                if df[col].dtype == 'object':  # Only check string columns
                    matches = []
                    
                    for idx, value in df[col].items():
                        if pd.isna(value):
                            continue
                        
                        value_str = str(value)
                        found_matches = re.findall(pattern, value_str, re.IGNORECASE)
                        
                        if found_matches:
                            matches.extend([
                                {
                                    "row": int(idx),
                                    "value": match if isinstance(match, str) else str(match),
                                    "context": value_str[:50] + "..." if len(value_str) > 50 else value_str
                                }
                                for match in found_matches
                            ])
                    
                    if matches:
                        pattern_findings[col] = {
                            "count": len(matches),
                            "samples": matches[:5],  # First 5 matches
                            "total_rows_affected": len(set(m["row"] for m in matches))
                        }
                        total_matches += len(matches)
            
            if pattern_findings:
                findings[pattern_name] = pattern_findings
        
        # Determine risk level
        if total_matches == 0:
            risk_level = "none"
        elif total_matches < 10:
            risk_level = "low"
        elif total_matches < 100:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "risk_level": risk_level,
            "total_matches": total_matches,
            "patterns_found": list(findings.keys()),
            "findings": findings,
            "summary": {
                "columns_affected": len(set(
                    col for pattern_findings in findings.values() 
                    for col in pattern_findings.keys()
                )),
                "rows_affected": len(set(
                    match["row"] for pattern_findings in findings.values()
                    for col_findings in pattern_findings.values()
                    for match in col_findings["samples"]
                ))
            }
        }
    
    def redact(
        self,
        df_ref: str,
        patterns: Optional[List[str]] = None,
        replacement: str = "[REDACTED]"
    ) -> Dict[str, Any]:
        """
        Redact PII from DataFrame.
        
        Args:
            df_ref: Reference to DataFrame
            patterns: List of pattern names to redact (default: all)
            replacement: Replacement string for PII
            
        Returns:
            Reference to redacted DataFrame and redaction summary
        """
        df = load_pickle(df_ref)
        patterns = patterns or list(self.patterns.keys())
        
        df_redacted = df.copy()
        redaction_log = {}
        total_redactions = 0
        
        for pattern_name in patterns:
            if pattern_name not in self.patterns:
                continue
            
            pattern = self.patterns[pattern_name]
            pattern_redactions = {}
            
            # Redact each column
            for col in df.columns:
                if df[col].dtype == 'object':  # Only process string columns
                    redactions_in_col = 0
                    
                    def redact_match(match):
                        nonlocal redactions_in_col
                        redactions_in_col += 1
                        return replacement
                    
                    # Apply redaction
                    df_redacted[col] = df_redacted[col].astype(str).str.replace(
                        pattern, redact_match, regex=True, case=False
                    )
                    
                    if redactions_in_col > 0:
                        pattern_redactions[col] = redactions_in_col
                        total_redactions += redactions_in_col
            
            if pattern_redactions:
                redaction_log[pattern_name] = pattern_redactions
        
        # Save redacted DataFrame
        from ..utils import hash_dataset
        dataset_hash = hash_dataset(df_redacted, "redacted")
        redacted_path = self.data_dir / f"redacted_{dataset_hash[:8]}.pkl"
        save_pickle(df_redacted, redacted_path)
        
        return {
            "df_ref_sanitized": str(redacted_path),
            "total_redactions": total_redactions,
            "redaction_log": redaction_log,
            "summary": {
                "patterns_redacted": list(redaction_log.keys()),
                "columns_affected": len(set(
                    col for pattern_redactions in redaction_log.values()
                    for col in pattern_redactions.keys()
                )),
                "replacement_string": replacement
            }
        }
    
    def check_column_names(self, df_ref: str) -> Dict[str, Any]:
        """
        Check column names for potential PII indicators.
        
        Args:
            df_ref: Reference to DataFrame
            
        Returns:
            Column name analysis
        """
        df = load_pickle(df_ref)
        
        suspicious_keywords = [
            'email', 'mail', 'phone', 'tel', 'ssn', 'social', 'security',
            'credit', 'card', 'account', 'password', 'pass', 'secret',
            'name', 'first', 'last', 'fname', 'lname', 'address', 'addr',
            'street', 'city', 'zip', 'postal', 'dob', 'birth', 'age'
        ]
        
        suspicious_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in suspicious_keywords:
                if keyword in col_lower:
                    suspicious_columns.append({
                        "column": col,
                        "keyword": keyword,
                        "risk_level": "high" if keyword in ['ssn', 'social', 'password', 'secret'] else "medium"
                    })
                    break
        
        return {
            "suspicious_columns": suspicious_columns,
            "total_suspicious": len(suspicious_columns),
            "high_risk_columns": [
                col["column"] for col in suspicious_columns 
                if col["risk_level"] == "high"
            ]
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "PII_scan",
                    "description": "Scan DataFrame for PII patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df_ref": {
                                "type": "string",
                                "description": "Reference to pickled DataFrame"
                            },
                            "patterns": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["email", "phone", "ssn", "credit_card", "ip_address", "url"]
                                },
                                "description": "List of PII patterns to check (default: all)"
                            }
                        },
                        "required": ["df_ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "PII_redact",
                    "description": "Redact PII from DataFrame using pattern matching",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df_ref": {
                                "type": "string",
                                "description": "Reference to pickled DataFrame"
                            },
                            "patterns": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["email", "phone", "ssn", "credit_card", "ip_address", "url"]
                                },
                                "description": "List of PII patterns to redact (default: all)"
                            },
                            "replacement": {
                                "type": "string",
                                "description": "Replacement string for PII (default: [REDACTED])"
                            }
                        },
                        "required": ["df_ref"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "PII_check_column_names",
                    "description": "Check column names for potential PII indicators",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "df_ref": {
                                "type": "string",
                                "description": "Reference to pickled DataFrame"
                            }
                        },
                        "required": ["df_ref"]
                    }
                }
            }
        ]
