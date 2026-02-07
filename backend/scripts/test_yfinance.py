#!/usr/bin/env python3
"""
Quick test script to verify Yahoo Finance integration

Usage:
    python scripts/test_yfinance.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_yfinance_installation():
    """Test if yfinance is installed"""
    print("=" * 60)
    print("Testing yfinance installation...")
    print("=" * 60)
    
    try:
        import yfinance as yf
        print("✅ yfinance is installed")
        print(f"   Version: {yf.__version__ if hasattr(yf, '__version__') else 'unknown'}")
        return True
    except ImportError:
        print("❌ yfinance is NOT installed")
        print("\nInstall with: pip install yfinance")
        return False


def test_fetch_sample_data():
    """Test fetching sample data from Yahoo Finance"""
    print("\n" + "=" * 60)
    print("Testing Yahoo Finance data fetch...")
    print("=" * 60)
    
    try:
        import yfinance as yf
        
        # Test with State Bank of India
        print("\nFetching State Bank of India (SBIN.NS)...")
        ticker = yf.Ticker("SBIN.NS")
        hist = ticker.history(period="1mo")
        
        if not hist.empty:
            print(f"✅ Successfully fetched {len(hist)} days of data")
            print(f"   Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
            print(f"   Latest close: ₹{hist['Close'].iloc[-1]:.2f}")
            
            # Compute returns
            returns = hist['Close'].pct_change().dropna()
            print(f"   Mean daily return: {returns.mean():.4%}")
            print(f"   Volatility: {returns.std():.4%}")
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return False


def test_network_builder():
    """Test network builder with Yahoo Finance"""
    print("\n" + "=" * 60)
    print("Testing Network Builder with Yahoo Finance...")
    print("=" * 60)
    
    try:
        from app.ml.data.network_builder import CompositeNetworkBuilder
        
        builder = CompositeNetworkBuilder('app/ml/data')
        
        # Test with a few major banks
        test_banks = [
            'STATE BANK OF INDIA',
            'HDFC BANK LTD.',
            'ICICI BANK LIMITED',
            'AXIS BANK LIMITED'
        ]
        
        print(f"\nTesting with {len(test_banks)} banks:")
        for bank in test_banks:
            print(f"  - {bank}")
        
        print("\nFetching market data...")
        bank_names, corr_matrix = builder.build_market_channel(
            test_banks,
            use_synthetic=False
        )
        
        if corr_matrix is not None:
            print(f"✅ Market channel built successfully")
            print(f"   Banks processed: {len(bank_names)}")
            print(f"   Correlation matrix: {corr_matrix.shape}")
            print(f"   Mean correlation: {corr_matrix.mean():.4f}")
            
            # Show correlation pairs
            print("\n   Sample correlations:")
            import numpy as np
            for i in range(min(3, len(bank_names))):
                for j in range(i+1, min(3, len(bank_names))):
                    print(f"     {bank_names[i][:20]:20s} <-> {bank_names[j][:20]:20s}: {corr_matrix[i,j]:.4f}")
            
            return True
        else:
            print("❌ Failed to build market channel")
            return False
            
    except Exception as e:
        print(f"❌ Error in network builder: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  Yahoo Finance Integration Test Suite".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    results = []
    
    # Test 1: Installation
    results.append(("yfinance installation", test_yfinance_installation()))
    
    if results[0][1]:  # Only continue if yfinance is installed
        # Test 2: Sample data fetch
        results.append(("Sample data fetch", test_fetch_sample_data()))
        
        # Test 3: Network builder integration
        results.append(("Network builder", test_network_builder()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Yahoo Finance integration is working.")
    else:
        print("⚠️ Some tests failed. Check errors above.")
        if not results[0][1]:
            print("\n💡 Install yfinance with: pip install yfinance")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
