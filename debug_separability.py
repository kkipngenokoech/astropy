#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Let's try to import just the separable module directly
try:
    from astropy.modeling.separable import _separable, _cstack, _coord_matrix, separability_matrix
    from astropy.modeling.core import Model, CompoundModel
    print("Successfully imported separable functions")
    
    # Create mock models to test
    class MockModel(Model):
        def __init__(self, n_inputs, n_outputs, separable=True, name="mock"):
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
            self.separable = separable
            self.name = name
            
        def _calculate_separability_matrix(self):
            return NotImplemented
    
    # Test the issue
    linear1 = MockModel(1, 1, True, "linear1")
    linear2 = MockModel(1, 1, True, "linear2") 
    pix2sky = MockModel(2, 2, False, "pix2sky")
    
    # Create compound model manually
    cm = CompoundModel('&', linear1, linear2, name="cm")
    nested = CompoundModel('&', pix2sky, cm, name="nested")
    
    print(f"cm n_inputs: {cm.n_inputs}, n_outputs: {cm.n_outputs}")
    print(f"nested n_inputs: {nested.n_inputs}, n_outputs: {nested.n_outputs}")
    
    # Test separability
    cm_sep = separability_matrix(cm)
    print("CM separability matrix:")
    print(cm_sep)
    
    nested_sep = separability_matrix(nested)
    print("Nested separability matrix:")
    print(nested_sep)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()