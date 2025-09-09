#!/usr/bin/env python3
"""
Test script to verify that seeding is working correctly in our electrode analysis.
This will run a minimal version of the analysis twice and check for identical results.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 42

def test_reproducibility():
    """Test that our seeding produces reproducible results"""
    
    # Create some sample data
    np.random.seed(RANDOM_SEED)
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 3, 100)  # 3 classes
    
    results_run1 = []
    results_run2 = []
    
    # Run 1
    print("🔄 Run 1...")
    np.random.seed(RANDOM_SEED)
    
    rf1 = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=1  # Use single job for deterministic results
    )
    
    cv1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores1 = cross_val_score(rf1, X, y, cv=cv1, scoring="accuracy")
    results_run1 = scores1.tolist()
    
    # Run 2  
    print("🔄 Run 2...")
    np.random.seed(RANDOM_SEED)
    
    rf2 = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=1  # Use single job for deterministic results
    )
    
    cv2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores2 = cross_val_score(rf2, X, y, cv=cv2, scoring="accuracy")
    results_run2 = scores2.tolist()
    
    # Compare results
    print(f"\n📊 Results Comparison:")
    print(f"Run 1 scores: {results_run1}")
    print(f"Run 2 scores: {results_run2}")
    
    # Check if identical
    identical = np.allclose(results_run1, results_run2, atol=1e-10)
    
    if identical:
        print("✅ SUCCESS: Results are identical - seeding is working!")
        print(f"   Mean accuracy: {np.mean(results_run1):.6f}")
        print(f"   Std accuracy: {np.std(results_run1):.6f}")
    else:
        print("❌ FAILURE: Results differ - seeding may not be working properly")
        print(f"   Difference: {np.array(results_run1) - np.array(results_run2)}")
    
    return identical

if __name__ == "__main__":
    print("🌱 Testing Random Seed Reproducibility")
    print("=" * 50)
    
    # Test basic numpy seeding
    np.random.seed(RANDOM_SEED)
    rand1 = np.random.rand(5)
    
    np.random.seed(RANDOM_SEED)  
    rand2 = np.random.rand(5)
    
    print(f"NumPy seed test:")
    print(f"  First run:  {rand1}")
    print(f"  Second run: {rand2}")
    print(f"  Identical:  {np.array_equal(rand1, rand2)}")
    print()
    
    # Test sklearn reproducibility
    success = test_reproducibility()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All seeding tests passed! Your analysis will be reproducible.")
    else:
        print("⚠️  Seeding tests failed. Check your random state settings.")
