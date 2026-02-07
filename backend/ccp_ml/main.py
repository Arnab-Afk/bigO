#!/usr/bin/env python3
"""
CCP Risk Analysis Runner

Main entry point to run the complete CCP-centric risk analysis pipeline.
Implements ML_Flow.md specification from scratch.

Usage:
    python -m ccp_ml.main
    python -m ccp_ml.main --year 2024 --include-market
    python -m ccp_ml.main --output results.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for CCP risk analysis."""
    parser = argparse.ArgumentParser(
        description='CCP Risk Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m ccp_ml.main                          # Run with defaults
    python -m ccp_ml.main --year 2024              # Analyze specific year
    python -m ccp_ml.main --include-market         # Include market data
    python -m ccp_ml.main --output results.json    # Save results to file
    python -m ccp_ml.main --no-train               # Skip model training
        """
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=None,
        help='Path to data directory (default: use package data)'
    )
    
    parser.add_argument(
        '--year', '-y',
        type=int,
        default=None,
        help='Specific year to analyze (default: all years)'
    )
    
    parser.add_argument(
        '--include-market',
        action='store_true',
        help='Include market data from Yahoo Finance'
    )
    
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Skip model training (use for inference only)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Import pipeline (after parsing args to get early errors)
        from . import CCPPipeline
        
        logger.info("="*70)
        logger.info("CCP RISK ANALYSIS - RUDRA Financial Infrastructure Risk Platform")
        logger.info("="*70)
        logger.info(f"Configuration:")
        logger.info(f"  Data directory: {args.data_dir or 'default'}")
        logger.info(f"  Target year: {args.year or 'all'}")
        logger.info(f"  Include market data: {args.include_market}")
        logger.info(f"  Train model: {not args.no_train}")
        logger.info("")
        
        # Create and run pipeline
        pipeline = CCPPipeline(data_dir=args.data_dir)
        
        results = pipeline.run(
            target_year=args.year,
            train=not args.no_train,
            include_market_data=args.include_market
        )
        
        # Display summary
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\n📊 Participants Analyzed: {results.get('n_participants', 'N/A')}")
        
        if 'risk_distribution' in results:
            print(f"\n📈 Risk Distribution:")
            for level, count in results['risk_distribution'].items():
                bar = '█' * min(count, 20)
                print(f"    {level:12s}: {count:3d} {bar}")
        
        if 'margin_summary' in results:
            ms = results['margin_summary']
            print(f"\n💰 Margin Summary:")
            print(f"    Total margin requirement: {ms.get('total_margin', 0):.4f}")
            print(f"    High-margin participants: {ms.get('high_margin_count', 0)}")
        
        if 'default_fund' in results:
            df = results['default_fund']
            print(f"\n🏦 Default Fund:")
            print(f"    Total size: {df.get('total_fund', 0):,.0f}")
            print(f"    Cover-N standard: {df.get('cover_n', 'N/A')}")
        
        if 'spectral_metrics' in results:
            sm = results['spectral_metrics']
            print(f"\n🔬 Systemic Risk Metrics:")
            print(f"    Spectral radius: {sm.get('spectral_radius', 0):.4f}")
            print(f"    Fiedler value: {sm.get('fiedler_value', 0):.4f}")
            print(f"    Amplification risk: {sm.get('amplification_risk', 'N/A')}")
            print(f"    Fragmentation risk: {sm.get('fragmentation_risk', 'N/A')}")
        
        if 'policies' in results:
            print(f"\n📋 Policy Recommendations ({len(results['policies'])}):")
            for i, policy in enumerate(results['policies'], 1):
                priority = policy.get('priority', 'unknown').upper()
                recommendation = policy.get('recommendation', 'N/A')
                print(f"    {i}. [{priority}] {recommendation}")
        
        # Export results if requested
        if args.output:
            pipeline.export_results(args.output)
            print(f"\n✅ Results exported to: {args.output}")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        return 0
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("  pip install pandas numpy networkx scikit-learn")
        return 1
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Please check the data directory path")
        return 1
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
