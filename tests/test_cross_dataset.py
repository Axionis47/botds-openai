"""Cross-dataset smoke tests."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from botds import Config, Pipeline


class TestCrossDataset(unittest.TestCase):
    """Test pipeline on all three datasets."""
    
    def setUp(self):
        """Set up test environment."""
        # Skip tests if no OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not set")
        
        # Create temporary directory for artifacts
        self.temp_dir = tempfile.mkdtemp()
    
    def test_iris_dataset(self):
        """Test pipeline on Iris dataset."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        
        # Override output directory
        config.report.out_dir = self.temp_dir
        
        # Run pipeline
        pipeline = Pipeline(config)
        result = pipeline.run()
        
        # Check success
        self.assertEqual(result["status"], "success")
        self.assertIn("job_id", result)
        
        # Check artifacts exist
        artifacts_dir = Path(result["artifacts_dir"])
        self.assertTrue(artifacts_dir.exists())
        
        # Check key files exist
        expected_files = [
            "handoffs/profile.json",
            "handoffs/split_indices.json", 
            "handoffs/feature_plan.json",
            "reports/one_pager.html",
            "logs/decision_log.jsonl",
            "run_manifest.json"
        ]
        
        for file_path in expected_files:
            full_path = artifacts_dir / file_path
            self.assertTrue(full_path.exists(), f"Missing file: {file_path}")
    
    def test_breast_cancer_dataset(self):
        """Test pipeline on Breast Cancer dataset."""
        config_path = Path(__file__).parent.parent / "configs" / "breast_cancer.yaml"
        config = Config.from_yaml(str(config_path))
        
        # Override output directory
        config.report.out_dir = self.temp_dir
        
        # Run pipeline
        pipeline = Pipeline(config)
        result = pipeline.run()
        
        # Check success
        self.assertEqual(result["status"], "success")
        
        # Check primary metric is pr_auc
        self.assertEqual(config.metrics["primary"], "pr_auc")
    
    def test_diabetes_dataset(self):
        """Test pipeline on Diabetes dataset (regression)."""
        config_path = Path(__file__).parent.parent / "configs" / "diabetes.yaml"
        config = Config.from_yaml(str(config_path))
        
        # Override output directory
        config.report.out_dir = self.temp_dir
        
        # Run pipeline
        pipeline = Pipeline(config)
        result = pipeline.run()
        
        # Check success
        self.assertEqual(result["status"], "success")
        
        # Check primary metric is mae
        self.assertEqual(config.metrics["primary"], "mae")
    
    def test_all_datasets_produce_reports(self):
        """Test that all datasets produce complete reports."""
        datasets = ["iris", "breast_cancer", "diabetes"]
        
        for dataset in datasets:
            with self.subTest(dataset=dataset):
                config_path = Path(__file__).parent.parent / "configs" / f"{dataset}.yaml"
                config = Config.from_yaml(str(config_path))
                config.report.out_dir = self.temp_dir
                
                pipeline = Pipeline(config)
                result = pipeline.run()
                
                self.assertEqual(result["status"], "success")
                
                # Check report files
                artifacts_dir = Path(result["artifacts_dir"])
                one_pager = artifacts_dir / "reports" / "one_pager.html"
                appendix = artifacts_dir / "reports" / "appendix.html"
                
                self.assertTrue(one_pager.exists())
                self.assertTrue(appendix.exists())
                
                # Check report content is not empty
                self.assertGreater(one_pager.stat().st_size, 1000)  # At least 1KB
                self.assertGreater(appendix.stat().st_size, 100)   # At least 100B


if __name__ == "__main__":
    unittest.main()
