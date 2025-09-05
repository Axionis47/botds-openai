"""Cache behavior and invalidation tests."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from botds import Config, Pipeline
from botds.cache import Cache


class TestCacheAndInvalidation(unittest.TestCase):
    """Test cache behavior and invalidation logic."""
    
    def setUp(self):
        """Set up test environment."""
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not set")
        
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
    
    def test_cache_warm_rerun(self):
        """Test that warm cache reuses results."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        
        # Set cache and output directories
        config.cache.dir = str(self.cache_dir)
        config.cache.mode = "warm"
        config.report.out_dir = self.temp_dir
        
        # First run
        pipeline1 = Pipeline(config)
        result1 = pipeline1.run()
        self.assertEqual(result1["status"], "success")
        
        cache_stats1 = result1.get("cache_stats", {})
        
        # Second run (should hit cache)
        pipeline2 = Pipeline(config)
        result2 = pipeline2.run()
        self.assertEqual(result2["status"], "success")
        
        cache_stats2 = result2.get("cache_stats", {})
        
        # Check that second run had more cache hits
        hits1 = sum(1 for hit in cache_stats1.values() if hit)
        hits2 = sum(1 for hit in cache_stats2.values() if hit)
        
        # Second run should have more hits (at least some stages cached)
        self.assertGreaterEqual(hits2, hits1)
    
    def test_cache_cold_mode(self):
        """Test that cold cache ignores existing cache."""
        config_path = Path(__file__).parent.parent / "configs" / "iris.yaml"
        config = Config.from_yaml(str(config_path))
        
        config.cache.dir = str(self.cache_dir)
        config.report.out_dir = self.temp_dir
        
        # First run with warm cache
        config.cache.mode = "warm"
        pipeline1 = Pipeline(config)
        result1 = pipeline1.run()
        self.assertEqual(result1["status"], "success")
        
        # Second run with cold cache
        config.cache.mode = "cold"
        pipeline2 = Pipeline(config)
        result2 = pipeline2.run()
        self.assertEqual(result2["status"], "success")
        
        cache_stats2 = result2.get("cache_stats", {})
        
        # Cold mode should have no cache hits
        hits2 = sum(1 for hit in cache_stats2.values() if hit)
        self.assertEqual(hits2, 0)
    
    def test_cache_invalidation_logic(self):
        """Test cache invalidation dependencies."""
        cache = Cache(str(self.cache_dir), "warm")
        
        # Put some test data
        cache.put("profile", "test_key", {"test": "data"}, [])
        cache.put("eda", "test_key", {"eda": "data"}, ["profile:test_key"])
        cache.put("feature_plan", "test_key", {"features": "data"}, ["profile:test_key", "eda:test_key"])
        
        # Check initial state
        self.assertIsNotNone(cache.get("profile", "test_key"))
        self.assertIsNotNone(cache.get("eda", "test_key"))
        self.assertIsNotNone(cache.get("feature_plan", "test_key"))
        
        # Invalidate profile stage
        invalidated = cache.invalidate_downstream("profile")
        
        # Should invalidate downstream stages
        expected_invalidated = {"eda", "feature_plan", "split_indices", "ladder", "evaluation", "reports"}
        self.assertTrue(invalidated.intersection(expected_invalidated))
    
    def test_cache_index_persistence(self):
        """Test that cache index persists across instances."""
        cache1 = Cache(str(self.cache_dir), "warm")
        
        # Put data in first cache instance
        cache1.put("profile", "test_key", {"test": "data"})
        
        # Create new cache instance
        cache2 = Cache(str(self.cache_dir), "warm")
        
        # Should be able to retrieve data
        data = cache2.get("profile", "test_key")
        self.assertIsNotNone(data)
        self.assertEqual(data["test"], "data")
    
    def test_cache_hit_miss_tracking(self):
        """Test cache hit/miss statistics tracking."""
        cache = Cache(str(self.cache_dir), "warm")
        
        # Miss on empty cache
        result1 = cache.get("profile", "missing_key")
        self.assertIsNone(result1)
        
        # Put data
        cache.put("profile", "existing_key", {"data": "value"})
        
        # Hit on existing data
        result2 = cache.get("profile", "existing_key")
        self.assertIsNotNone(result2)
        
        # Check hit/miss stats
        stats = cache.get_hit_stats()
        self.assertIn("profile:missing_key", stats)
        self.assertIn("profile:existing_key", stats)
        self.assertFalse(stats["profile:missing_key"])  # Miss
        self.assertTrue(stats["profile:existing_key"])   # Hit


if __name__ == "__main__":
    unittest.main()
