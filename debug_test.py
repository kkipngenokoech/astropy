#!/usr/bin/env python

import sys
sys.path.insert(0, '.')

import numpy as np

# Mock the necessary classes for testing
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
from astropy.modeling.separable import _separable, _cstack, _coord_matrix

# Create test models
linear1 = MockModel(1, 1, separable=True)  # Linear1D(10)
linear2 = MockModel(1, 1, separable=True)  # Linear1D(5)
pix2sky = MockModel(2, 2, separable=False)  # Pix2Sky_TAN()

# Create compound model: Linear1D(10) & Linear1D(5)
cm = MockCompoundModel('&', linear1, linear2)

# Create nested compound model: Pix2Sky_TAN() & (Linear1D(10) & Linear1D(5))
nested = MockCompoundModel('&', pix2sky, cm)

print("Testing separability calculation...")

# Test the basic compound model
print("Basic compound model (Linear1D & Linear1D):")
cm_matrix = _separable(cm)
print("Result shape:", cm_matrix.shape)
print("Result:\n", cm_matrix)

# Test the nested compound model
print("\nNested compound model (Pix2Sky_TAN & (Linear1D & Linear1D)):")
nested_matrix = _separable(nested)
print("Result shape:", nested_matrix.shape)
print("Result:\n", nested_matrix)

# Expected result for nested model should be:
expected = np.array([
    [1, 1, 0, 0],  # First TAN output depends on first 2 inputs
    [1, 1, 0, 0],  # Second TAN output depends on first 2 inputs  
    [0, 0, 1, 0],  # First Linear1D output depends on 3rd input
    [0, 0, 0, 1]   # Second Linear1D output depends on 4th input
])
print("\nExpected:\n", expected)
print("Match expected?", np.array_equal(nested_matrix, expected))