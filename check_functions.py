import importlib.util

# Load the module without executing main
spec = importlib.util.spec_from_file_location('test_module', 'indoor_electrode_analysis.py')
module = importlib.util.module_from_spec(spec)

import sys
sys.modules['test_module'] = module
spec.loader.exec_module(module)

# Check if new functions exist
functions_to_check = [
    'train_and_evaluate_rf_with_models',
    'test_models_on_outdoor_data', 
    'compare_8vs4_electrodes_with_ttest_and_outdoor_testing',
    'create_enhanced_comparison_visualization'
]

all_found = True
for func_name in functions_to_check:
    if hasattr(module, func_name):
        print(f'✅ {func_name} - defined')
    else:
        print(f'❌ {func_name} - missing')
        all_found = False

if all_found:
    print('✅ All enhanced functions are properly defined!')
else:
    print('❌ Some functions are missing!')
