"""Artifact storage and handoff ledger tools."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..context import HandoffLedger as BaseHandoffLedger
from ..utils import ensure_dir, save_json


class ArtifactStore:
    """Artifact storage and management."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.reports_dir = ensure_dir(artifacts_dir / "reports")
        self.tables_dir = ensure_dir(artifacts_dir / "tables")
        self.figures_dir = ensure_dir(artifacts_dir / "figures")
    
    def write_report(
        self,
        kind: str,
        payload: Dict[str, Any],
        format: str = "html"
    ) -> Dict[str, Any]:
        """
        Write report (one-pager or appendix).
        
        Args:
            kind: 'one_pager' or 'appendix'
            payload: Report content payload
            format: 'html' or 'md'
            
        Returns:
            Report file reference
        """
        if kind not in ["one_pager", "appendix"]:
            raise ValueError(f"Unknown report kind: {kind}")
        
        if format == "html":
            report_content = self._generate_html_report(kind, payload)
            file_ext = "html"
        else:
            report_content = self._generate_markdown_report(kind, payload)
            file_ext = "md"
        
        # Save report
        report_path = self.reports_dir / f"{kind}.{file_ext}"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        return {
            "report_ref": str(report_path),
            "kind": kind,
            "format": format,
            "size_bytes": len(report_content)
        }
    
    def _generate_html_report(self, kind: str, payload: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        if kind == "one_pager":
            return self._generate_one_pager_html(payload)
        else:
            return self._generate_appendix_html(payload)
    
    def _generate_markdown_report(self, kind: str, payload: Dict[str, Any]) -> str:
        """Generate Markdown report content."""
        if kind == "one_pager":
            return self._generate_one_pager_md(payload)
        else:
            return self._generate_appendix_md(payload)
    
    def _generate_one_pager_html(self, payload: Dict[str, Any]) -> str:
        """Generate one-pager HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Executive One-Pager</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .success {{ background: #d4edda; border-left: 4px solid #28a745; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Executive One-Pager</h1>
    
    <h2>1. Problem & Success Metric</h2>
    <p><strong>Business Goal:</strong> {payload.get('business_goal', 'Not specified')}</p>
    <p><strong>Primary Metric:</strong> {payload.get('primary_metric', 'Not specified')}</p>
    
    <h2>2. Data Snapshot</h2>
    <div class="metric">
        <strong>Dataset:</strong> {payload.get('dataset_info', {}).get('shape', 'Unknown')} rows × columns<br>
        <strong>Target:</strong> {payload.get('target', 'Not specified')}<br>
        <strong>Leakage Status:</strong> {payload.get('leakage_status', 'Not checked')}
    </div>
    
    <h2>3. Top 3 Insights</h2>
    <ol>
"""
        
        insights = payload.get('top_insights', ['No insights available'] * 3)
        for insight in insights[:3]:
            html += f"        <li>{insight}</li>\n"
        
        html += """    </ol>
    
    <h2>4. Model Decision</h2>
    <div class="success">
        <strong>Selected Model:</strong> {model_name}<br>
        <strong>Performance:</strong> {performance}<br>
        <strong>Rationale:</strong> {rationale}
    </div>
    
    <h3>Mini-Leaderboard (Top 3)</h3>
    <table>
        <tr><th>Rank</th><th>Model</th><th>Score</th><th>CI</th><th>Runtime</th></tr>
""".format(
            model_name=payload.get('selected_model', {}).get('name', 'Not selected'),
            performance=payload.get('selected_model', {}).get('score', 'Not available'),
            rationale=payload.get('selected_model', {}).get('rationale', 'Not provided')
        )
        
        leaderboard = payload.get('leaderboard', [])
        for i, model in enumerate(leaderboard[:3], 1):
            html += f"""        <tr>
            <td>{i}</td>
            <td>{model.get('name', f'Model {i}')}</td>
            <td>{model.get('score', 'N/A')}</td>
            <td>{model.get('ci', 'N/A')}</td>
            <td>{model.get('runtime', 'N/A')}</td>
        </tr>
"""
        
        html += f"""    </table>
    
    <h2>5. Operating Point</h2>
    <div class="metric">
        <strong>Recommended Threshold:</strong> {payload.get('operating_point', {}).get('threshold', 'Not set')}<br>
        <strong>Conservative Alternative:</strong> {payload.get('operating_point', {}).get('conservative', 'Not set')}<br>
        <strong>Business Trade-off:</strong> {payload.get('operating_point', {}).get('tradeoff', 'Not specified')}
    </div>
    
    <h2>6. Robustness Grade</h2>
    <div class="{'success' if payload.get('robustness_grade', 'D') in ['A', 'B'] else 'warning'}">
        <strong>Grade:</strong> {payload.get('robustness_grade', 'Not assessed')}<br>
        <strong>Reason:</strong> {payload.get('robustness_reason', 'Not provided')}
    </div>
    
    <h2>7. Next Steps</h2>
    <ul>
"""
        
        next_steps = payload.get('next_steps', ['No next steps defined'])
        for step in next_steps:
            html += f"        <li>{step}</li>\n"
        
        html += """    </ul>
"""
        
        # Add shortcuts section if any
        shortcuts = payload.get('shortcuts_taken', [])
        if shortcuts:
            html += """    
    <h2>8. Run Assumptions & Shortcuts</h2>
    <div class="warning">
        <ul>
"""
            for shortcut in shortcuts:
                html += f"            <li>{shortcut}</li>\n"
            html += """        </ul>
    </div>
"""
        
        html += """
</body>
</html>"""
        
        return html
    
    def _generate_one_pager_md(self, payload: Dict[str, Any]) -> str:
        """Generate one-pager Markdown."""
        md = f"""# Executive One-Pager

## 1. Problem & Success Metric

**Business Goal:** {payload.get('business_goal', 'Not specified')}
**Primary Metric:** {payload.get('primary_metric', 'Not specified')}

## 2. Data Snapshot

- **Dataset:** {payload.get('dataset_info', {}).get('shape', 'Unknown')} rows × columns
- **Target:** {payload.get('target', 'Not specified')}
- **Leakage Status:** {payload.get('leakage_status', 'Not checked')}

## 3. Top 3 Insights

"""
        
        insights = payload.get('top_insights', ['No insights available'] * 3)
        for i, insight in enumerate(insights[:3], 1):
            md += f"{i}. {insight}\n"
        
        md += f"""
## 4. Model Decision

**Selected Model:** {payload.get('selected_model', {}).get('name', 'Not selected')}
**Performance:** {payload.get('selected_model', {}).get('score', 'Not available')}
**Rationale:** {payload.get('selected_model', {}).get('rationale', 'Not provided')}

### Mini-Leaderboard (Top 3)

| Rank | Model | Score | CI | Runtime |
|------|-------|-------|----|---------| 
"""
        
        leaderboard = payload.get('leaderboard', [])
        for i, model in enumerate(leaderboard[:3], 1):
            md += f"| {i} | {model.get('name', f'Model {i}')} | {model.get('score', 'N/A')} | {model.get('ci', 'N/A')} | {model.get('runtime', 'N/A')} |\n"
        
        md += f"""
## 5. Operating Point

- **Recommended Threshold:** {payload.get('operating_point', {}).get('threshold', 'Not set')}
- **Conservative Alternative:** {payload.get('operating_point', {}).get('conservative', 'Not set')}
- **Business Trade-off:** {payload.get('operating_point', {}).get('tradeoff', 'Not specified')}

## 6. Robustness Grade

**Grade:** {payload.get('robustness_grade', 'Not assessed')}
**Reason:** {payload.get('robustness_reason', 'Not provided')}

## 7. Next Steps

"""
        
        next_steps = payload.get('next_steps', ['No next steps defined'])
        for step in next_steps:
            md += f"- {step}\n"
        
        # Add shortcuts section if any
        shortcuts = payload.get('shortcuts_taken', [])
        if shortcuts:
            md += "\n## 8. Run Assumptions & Shortcuts\n\n"
            for shortcut in shortcuts:
                md += f"- {shortcut}\n"
        
        return md
    
    def _generate_appendix_html(self, payload: Dict[str, Any]) -> str:
        """Generate appendix HTML (simplified for MVP)."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Technical Appendix</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; }}
        .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>Technical Appendix</h1>
    
    <div class="section">
        <h2>Configuration</h2>
        <pre>{payload.get('config_snapshot', 'Not available')}</pre>
    </div>
    
    <div class="section">
        <h2>Reproducibility</h2>
        <p><strong>Dataset Hash:</strong> {payload.get('dataset_hash', 'Not available')}</p>
        <p><strong>Seeds:</strong> {payload.get('seeds', 'Not available')}</p>
    </div>
    
</body>
</html>"""
    
    def _generate_appendix_md(self, payload: Dict[str, Any]) -> str:
        """Generate appendix Markdown (simplified for MVP)."""
        return f"""# Technical Appendix

## Configuration

```
{payload.get('config_snapshot', 'Not available')}
```

## Reproducibility

- **Dataset Hash:** {payload.get('dataset_hash', 'Not available')}
- **Seeds:** {payload.get('seeds', 'Not available')}
"""
    
    def write_table(self, name: str, obj: Any) -> Dict[str, Any]:
        """Write table object to file."""
        table_path = self.tables_dir / f"{name}.json"
        hash_value = save_json(obj, table_path)
        
        return {
            "table_ref": str(table_path),
            "hash": hash_value
        }
    
    def write_fig(self, name: str, fig_ref: str) -> Dict[str, Any]:
        """Copy figure to figures directory."""
        import shutil
        
        source_path = Path(fig_ref)
        if not source_path.exists():
            raise FileNotFoundError(f"Figure not found: {fig_ref}")
        
        dest_path = self.figures_dir / f"{name}{source_path.suffix}"
        shutil.copy2(source_path, dest_path)
        
        return {
            "fig_ref": str(dest_path)
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "ArtifactStore_write_report",
                    "description": "Write one-pager or appendix report",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["one_pager", "appendix"],
                                "description": "Type of report to generate"
                            },
                            "payload": {
                                "type": "object",
                                "description": "Report content payload"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["html", "md"],
                                "description": "Report format (default: html)"
                            }
                        },
                        "required": ["kind", "payload"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ArtifactStore_write_table",
                    "description": "Write table object to JSON file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Table name"
                            },
                            "obj": {
                                "type": "object",
                                "description": "Table object to save"
                            }
                        },
                        "required": ["name", "obj"]
                    }
                }
            }
        ]


class HandoffLedger:
    """Wrapper for handoff ledger functionality."""
    
    def __init__(self, artifacts_dir: Path):
        self.ledger = BaseHandoffLedger(artifacts_dir / "logs" / "handoff_ledger.jsonl")
    
    def append(
        self,
        job_id: str,
        stage: str,
        input_refs: List[str],
        output_refs: List[str],
        schema_uri: str,
        hash_value: str
    ) -> Dict[str, Any]:
        """Append handoff entry."""
        self.ledger.append(job_id, stage, input_refs, output_refs, schema_uri, hash_value)
        
        return {
            "status": "logged",
            "stage": stage,
            "outputs": len(output_refs)
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "HandoffLedger_append",
                    "description": "Log a handoff between pipeline stages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "job_id": {
                                "type": "string",
                                "description": "Job ID"
                            },
                            "stage": {
                                "type": "string",
                                "description": "Pipeline stage name"
                            },
                            "input_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Input file references"
                            },
                            "output_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Output file references"
                            },
                            "schema_uri": {
                                "type": "string",
                                "description": "Schema URI for validation"
                            },
                            "hash_value": {
                                "type": "string",
                                "description": "Hash of the handoff data"
                            }
                        },
                        "required": ["job_id", "stage", "input_refs", "output_refs", "schema_uri", "hash_value"]
                    }
                }
            }
        ]
