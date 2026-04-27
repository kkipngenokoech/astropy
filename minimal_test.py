#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Mock the dependencies we need
import numpy as np

# Create minimal mock classes to test the separable functions
class MockModel:
    def __init__(self, n_inputs, n_outputs, separable=True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.separable = separable
    
    def _calculate_separability_matrix(self):
        return NotImplemented

class MockCompoundModel(MockModel):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        if op == '&':
            self.n_inputs = left.n_inputs + right.n_inputs
            self.n_outputs = left.n_outputs + right.n_outputs
        self.separable = True
    
    def _calculate_separability_matrix(self):
        return NotImplemented

# Import the functions we need to test
from astropy.modeling.separable import _separable, _cstack

# Test the fix
def test_cstack_fix():
    print("Testing _cstack function fix...")
    
    # Create a mock separability matrix for a compound model (2x2 identity)
    right_matrix = np.array([[1, 0], [0, 1]], dtype=float)
    
    # Create a mock separability matrix for a simple model (1x1)
    left_matrix = np.array([[1]], dtype=float)
    
    # Test the _cstack function
    result = _cstack(left_matrix, right_matrix)
    
    print("Left matrix (1x1):")
    print(left_matrix)
    print("Right matrix (2x2):")
    print(right_matrix)
    print("Result from _cstack:")
    print(result)
    
    # Expected result should be:
    # [[1, 0, 0],
    #  [0, 1, 0], 
    #  [0, 0, 1]]
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    print("Expected:")
    print(expected)
    
    if np.array_equal(result, expected):
        print("✓ Test passed!")
        return True
    else:
        print("✗ Test failed!")
        return False

def test_nested_compound_model():
    print("\nTesting nested compound model separability...")
    
    # Mock Pix2Sky_TAN (2 inputs, 2 outputs, not separable)
    pix2sky = MockModel(2, 2, separable=False)
    
    # Mock Linear1D models (1 input, 1 output, separable)
    linear1 = MockModel(1, 1, separable=True)
    linear2 = MockModel(1, 1, separable=True)
    
    # Create compound model: Linear1D & Linear1D
    cm = MockCompoundModel('&', linear1, linear2)
    
    # Create nested compound model: Pix2Sky_TAN & (Linear1D & Linear1D)
    nested = MockCompoundModel('&', pix2sky, cm)
    
    # Test separability
    result = _separable(nested)
    print("Nested model separability matrix:")
    print(result)
    
    # Expected: 
    # - Outputs 0,1 depend on inputs 0,1 (Pix2Sky_TAN)
    # - Output 2 depends only on input 2 (first Linear1D)
    # - Output 3 depends only on input 3 (second Linear1D)
    expected = np.array([
        [1, 1, 0, 0],  # Pix2Sky_TAN output 0
        [1, 1, 0, 0],  # Pix2Sky_TAN output 1
        [0, 0, 1, 0],  # Linear1D output 0
        [0, 0, 0, 1]   # Linear1D output 1
    ], dtype=float)
    
    print("Expected:")
    print(expected)
    
    if np.array_equal(result, expected):
        print("✓ Nested model test passed!")
        return True
    else:
        print("✗ Nested model test failed!")
        return False

if __name__ == "__main__":
    success1 = test_cstack_fix()
    success2 = test_nested_compound_model()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)