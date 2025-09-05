#!/usr/bin/env python3
"""Quick system test to verify the bot data scientist works."""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from botds import Config, Pipeline


def test_iris():
    """Test the system with Iris dataset."""
    print("Testing Bot Data Scientist with Iris dataset...")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it to test the system.")
        print("   export OPENAI_API_KEY=sk-your-key-here")
        return False
    
    try:
        # Load config
        config = Config.from_yaml("configs/iris.yaml")
        config.validate_environment()
        
        # Create pipeline
        pipeline = Pipeline(config)
        
        # Run pipeline
        print("Running pipeline...")
        result = pipeline.run()
        
        if result["status"] == "success":
            print(f"✅ Pipeline completed successfully!")
            print(f"📁 Artifacts: {result['artifacts_dir']}")
            
            # Check key files exist
            artifacts_dir = Path(result["artifacts_dir"])
            key_files = [
                "reports/one_pager.html",
                "logs/decision_log.jsonl",
                "run_manifest.json"
            ]
            
            for file_path in key_files:
                full_path = artifacts_dir / file_path
                if full_path.exists():
                    print(f"✅ {file_path}")
                else:
                    print(f"❌ Missing: {file_path}")
            
            return True
        else:
            print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Bot Data Scientist System Test")
    print("=" * 40)
    
    success = test_iris()
    
    if success:
        print("\n🎉 System test passed! The bot data scientist is working correctly.")
        print("\nNext steps:")
        print("1. Try other datasets: python -m cli.run --config configs/breast_cancer.yaml")
        print("2. Run full test suite: python -m pytest tests/")
        print("3. Check the generated reports in the artifacts directory")
    else:
        print("\n💥 System test failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure OPENAI_API_KEY is set correctly")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. See RUNBOOK.md for common issues")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
