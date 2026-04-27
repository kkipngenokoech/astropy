#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np

# Test the fixed _cstack function
def _cstack(left, right):
    """
    Function corresponding to '&' operation.

    Parameters
    ----------
    left, right : `astropy.modeling.Model` or ndarray
        If input is of an array, it is the output of `coord_matrix`.

    Returns
    -------
    result : ndarray
        Result from this operation.

    """
    # Determine the dimensions of left and right operands
    if hasattr(left, 'n_outputs'):  # Model
        left_outputs = left.n_outputs
        left_inputs = left.n_inputs
    else:
        left_outputs, left_inputs = left.shape
        
    if hasattr(right, 'n_outputs'):  # Model
        right_outputs = right.n_outputs
        right_inputs = right.n_inputs
    else:
        right_outputs, right_inputs = right.shape
    
    # Total dimensions for the result
    total_outputs = left_outputs + right_outputs
    total_inputs = left_inputs + right_inputs
    
    # Create the result matrix
    result = np.zeros((total_outputs, total_inputs))
    
    # Handle left operand
    if hasattr(left, 'n_outputs'):  # Model
        # Simulate _coord_matrix for a non-separable 2-input, 2-output model
        cleft = np.array([[1, 1], [1, 1]])  # Pix2Sky_TAN-like
        result[:left_outputs, :left_inputs] = cleft
    else:
        result[:left_outputs, :left_inputs] = left
    
    # Handle right operand
    if hasattr(right, 'n_outputs'):  # Model
        # This case won't happen in our test
        pass
    else:
        result[left_outputs:, left_inputs:] = right

    return result

# Simulate the separability matrix for Linear1D & Linear1D
linear_compound = np.array([[1, 0], [0, 1]])  # 2x2 diagonal matrix

# Simulate a model with 2 inputs, 2 outputs (like Pix2Sky_TAN)
class MockModel:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

pix2sky = MockModel(2, 2)

# Test the fixed _cstack implementation
result = _cstack(pix2sky, linear_compound)
print("Fixed _cstack result:")
print(result)
print()

# What we expect:
expected = np.array([
    [1, 1, 0, 0],  # output 0 depends on inputs 0,1
    [1, 1, 0, 0],  # output 1 depends on inputs 0,1  
    [0, 0, 1, 0],  # output 2 depends only on input 2
    [0, 0, 0, 1]   # output 3 depends only on input 3
])
print("Expected result:")
print(expected)
print()
print("Are they equal?", np.array_equal(result, expected))