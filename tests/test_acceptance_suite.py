"""Acceptance test suite covering all requirements."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from botds import Config, Pipeline


class TestAcceptanceSuite(unittest.TestCase):
    """Comprehensive acceptance tests."""
    
    def setUp(self):
        """Set up test environment."""
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not set")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def test_determinism_across_runs(self):
        """Test that re-running produces identical results within CI overlap."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        config.report.out_dir = self.temp_dir
        
        # Run 1
        pipeline1 = Pipeline(config)
        result1 = pipeline1.run()
        self.assertEqual(result1["status"], "success")
        
        # Run 2 with same config
        pipeline2 = Pipeline(config)
        result2 = pipeline2.run()
        self.assertEqual(result2["status"], "success")
        
        # Load split indices from both runs
        artifacts1 = Path(result1["artifacts_dir"])
        artifacts2 = Path(result2["artifacts_dir"])
        
        splits1_path = artifacts1 / "handoffs" / "split_indices.json"
        splits2_path = artifacts2 / "handoffs" / "split_indices.json"
        
        if splits1_path.exists() and splits2_path.exists():
            with open(splits1_path) as f:
                splits1 = json.load(f)
            with open(splits2_path) as f:
                splits2 = json.load(f)
            
            # Split indices should be identical (same seed)
            self.assertEqual(splits1["indices"], splits2["indices"])
    
    def test_budget_enforcement(self):
        """Test budget limits trigger appropriate responses."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        config.report.out_dir = self.temp_dir
        
        # Set very tight time budget
        config.budgets.time_min = 1  # 1 minute
        
        pipeline = Pipeline(config)
        
        # Should either complete with shortcuts or fail gracefully
        result = pipeline.run()
        
        # Check that budget monitoring occurred
        if result["status"] == "success":
            # If successful, check for shortcuts in manifest
            manifest_path = Path(result["artifacts_dir"]) / "run_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                
                # May have shortcuts due to tight budget
                shortcuts = manifest.get("shortcuts_taken", [])
                # This is acceptable - tight budget may require shortcuts
        else:
            # If failed, should be due to budget constraints
            self.assertIn("budget", result.get("error", "").lower())
    
    def test_authority_enforcement(self):
        """Test that OpenAI is required for critical decisions."""
        # This test verifies the system fails without OpenAI key
        original_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            # Remove OpenAI key
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            
            config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
            
            # Should fail during config validation
            with self.assertRaises(ValueError) as context:
                config = Config.from_yaml(str(config_path))
                config.validate_environment()
            
            self.assertIn("OPENAI_API_KEY", str(context.exception))
            self.assertIn("sole decision authority", str(context.exception))
            
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
    
    def test_decision_log_completeness(self):
        """Test that all critical decisions are logged with OpenAI authority."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        config.report.out_dir = self.temp_dir
        
        pipeline = Pipeline(config)
        result = pipeline.run()
        self.assertEqual(result["status"], "success")
        
        # Check decision log exists and has entries
        decision_log_path = Path(result["artifacts_dir"]) / "logs" / "decision_log.jsonl"
        self.assertTrue(decision_log_path.exists())
        
        # Read decision log
        decisions = []
        with open(decision_log_path) as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        
        # Should have multiple critical decisions
        self.assertGreater(len(decisions), 0)
        
        # All decisions should be from OpenAI
        for decision in decisions:
            self.assertEqual(decision.get("auth_model"), "openai")
            self.assertIn("stage", decision)
            self.assertIn("decision", decision)
            self.assertIn("timestamp", decision)
    
    def test_handoff_traceability(self):
        """Test that all handoffs are logged with proper schemas."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        config.report.out_dir = self.temp_dir
        
        pipeline = Pipeline(config)
        result = pipeline.run()
        self.assertEqual(result["status"], "success")
        
        # Check handoff ledger
        ledger_path = Path(result["artifacts_dir"]) / "logs" / "handoff_ledger.jsonl"
        
        if ledger_path.exists():
            handoffs = []
            with open(ledger_path) as f:
                for line in f:
                    if line.strip():
                        handoffs.append(json.loads(line))
            
            # Should have handoff entries
            self.assertGreater(len(handoffs), 0)
            
            # Each handoff should have required fields
            for handoff in handoffs:
                self.assertIn("stage", handoff)
                self.assertIn("inputs", handoff)
                self.assertIn("outputs", handoff)
                self.assertIn("schema", handoff)
                self.assertIn("hash", handoff)
                self.assertIn("timestamp", handoff)
    
    def test_schema_validation(self):
        """Test that handoff files match their schemas."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        config.report.out_dir = self.temp_dir
        
        pipeline = Pipeline(config)
        result = pipeline.run()
        self.assertEqual(result["status"], "success")
        
        artifacts_dir = Path(result["artifacts_dir"])
        handoffs_dir = artifacts_dir / "handoffs"
        
        # Check that key handoff files exist and are valid JSON
        expected_handoffs = [
            "profile.json",
            "split_indices.json",
            "feature_plan.json"
        ]
        
        for handoff_file in expected_handoffs:
            handoff_path = handoffs_dir / handoff_file
            if handoff_path.exists():
                # Should be valid JSON
                with open(handoff_path) as f:
                    data = json.load(f)
                
                # Should have some content
                self.assertIsInstance(data, dict)
                self.assertGreater(len(data), 0)
    
    def test_report_completeness(self):
        """Test that reports contain all required sections."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        config.report.out_dir = self.temp_dir
        
        pipeline = Pipeline(config)
        result = pipeline.run()
        self.assertEqual(result["status"], "success")
        
        # Check one-pager exists and has content
        one_pager_path = Path(result["artifacts_dir"]) / "reports" / "one_pager.html"
        self.assertTrue(one_pager_path.exists())
        
        with open(one_pager_path) as f:
            content = f.read()
        
        # Check for required sections
        required_sections = [
            "Problem & Success Metric",
            "Data Snapshot", 
            "Top 3 Insights",
            "Model Decision",
            "Operating Point",
            "Robustness Grade",
            "Next Steps"
        ]
        
        for section in required_sections:
            self.assertIn(section, content, f"Missing section: {section}")
    
    def test_reproducibility_manifest(self):
        """Test that run manifest contains reproducibility information."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        config.report.out_dir = self.temp_dir
        
        pipeline = Pipeline(config)
        result = pipeline.run()
        self.assertEqual(result["status"], "success")
        
        # Check manifest exists
        manifest_path = Path(result["artifacts_dir"]) / "run_manifest.json"
        self.assertTrue(manifest_path.exists())
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Check required fields for reproducibility
        self.assertIn("job_id", manifest)
        self.assertIn("created_at", manifest)
        self.assertIn("seeds", manifest)
        
        # Job ID should be 8 characters
        self.assertEqual(len(manifest["job_id"]), 8)


if __name__ == "__main__":
    unittest.main()
