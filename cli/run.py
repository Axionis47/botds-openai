"""CLI runner for Bot Data Scientist pipeline."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from botds import Config, Pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bot Data Scientist - OpenAI-led ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.run --config configs/iris.yaml
  python -m cli.run --config configs/breast_cancer.yaml
  python -m cli.run --config configs/diabetes.yaml
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = Config.from_yaml(args.config)
        
        # Validate environment (OpenAI API key)
        config.validate_environment()
        
        # Create and run pipeline
        pipeline = Pipeline(config)
        result = pipeline.run()
        
        if result["status"] == "success":
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📁 Artifacts written to: {result['artifacts_dir']}")
            
            # Show cache statistics
            cache_stats = result.get("cache_stats", {})
            if cache_stats:
                hits = sum(1 for hit in cache_stats.values() if hit)
                total = len(cache_stats)
                print(f"💾 Cache: {hits}/{total} hits")
            
            return 0
        else:
            print(f"\n❌ Pipeline failed: {result['error']}")
            print(f"📁 Partial artifacts in: {result['artifacts_dir']}")
            return 1
            
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
