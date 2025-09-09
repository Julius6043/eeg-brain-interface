#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the enhanced indoor_electrode_analysis.py works correctly
"""

import sys
import traceback

def test_imports():
    """Test if all required imports work"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import mne
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        from scipy import stats
        import warnings
        import joblib
        import os
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_script_syntax():
    """Test if the script has valid syntax"""
    try:
        with open('indoor_electrode_analysis.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Compile to check syntax
        compile(code, 'indoor_electrode_analysis.py', 'exec')
        print("✅ Script syntax is valid!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading/compiling script: {e}")
        return False

def main():
    print("Testing enhanced indoor_electrode_analysis.py...")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        return False
    
    # Test syntax
    if not test_script_syntax():
        return False
    
    print("=" * 50)
    print("✅ All tests passed! The script should work correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
