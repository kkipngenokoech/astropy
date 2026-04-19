import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np

# Mock the necessary classes and functions for testing
class Model:
    def __init__(self, n_inputs=1, n_outputs=1, separable=True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.separable = separable
    
    def _calculate_separability_matrix(self):
        return NotImplemented

class CompoundModel(Model):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        
        if op == '&':
            self.n_inputs = left.n_inputs + right.n_inputs
            self.n_outputs = left.n_outputs + right.n_outputs
        
    def _calculate_separability_matrix(self):
        return NotImplemented

# Import the functions we want to test
from astropy.modeling.separable import _separable, _cstack

# Test the fix
def test_nested_compound_models():
    # Create simple models
    pix2sky = Model(n_inputs=2, n_outputs=2, separable=False)  # Pix2Sky_TAN-like
    linear1 = Model(n_inputs=1, n_outputs=1, separable=True)   # Linear1D-like
    linear2 = Model(n_inputs=1, n_outputs=1, separable=True)   # Linear1D-like
    
    # Create compound model: linear1 & linear2
    cm = CompoundModel('&', linear1, linear2)
    
    # Test simple compound model separability
    cm_sep = _separable(cm)
    print("Simple compound model separability matrix:")
    print(cm_sep)
    
    # Expected: [[1, 0], [0, 1]] (diagonal)
    expected_cm = np.array([[1, 0], [0, 1]])
    assert np.array_equal(cm_sep, expected_cm), f"Expected {expected_cm}, got {cm_sep}"
    
    # Create nested compound model: pix2sky & cm
    nested = CompoundModel('&', pix2sky, cm)
    
    # Test nested compound model separability
    nested_sep = _separable(nested)
    print("Nested compound model separability matrix:")
    print(nested_sep)
    
    # Expected: outputs 0,1 depend on inputs 0,1; outputs 2,3 depend on inputs 2,3 respectively
    expected_nested = np.array([
        [1, 1, 0, 0],  # output 0 depends on inputs 0,1
        [1, 1, 0, 0],  # output 1 depends on inputs 0,1  
        [0, 0, 1, 0],  # output 2 depends only on input 2
        [0, 0, 0, 1]   # output 3 depends only on input 3
    ])
    print("Expected nested separability matrix:")
    print(expected_nested)
    
    assert np.array_equal(nested_sep, expected_nested), f"Expected {expected_nested}, got {nested_sep}"
    print("Test passed!")

if __name__ == "__main__":
    test_nested_compound_models()