#!/usr/bin/env python3

import numpy as np

# Simulate the separable.py functions without importing astropy
def _compute_n_outputs(left, right):
    if hasattr(left, 'n_outputs'):
        lnout = left.n_outputs
    else:
        lnout = left.shape[0]
    if hasattr(right, 'n_outputs'):
        rnout = right.n_outputs
    else:
        rnout = right.shape[0]
    return lnout + rnout

def _cstack(left, right):
    """Current implementation"""
    noutp = _compute_n_outputs(left, right)
    
    # Simulate what happens when left is a Model (Pix2Sky_TAN)
    # and right is an ndarray (result from compound Linear1D & Linear1D)
    if hasattr(left, 'n_outputs'):  # Model
        # Simulate _coord_matrix for a non-separable 2-input, 2-output model
        cleft = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    else:  # ndarray
        cleft = np.zeros((noutp, left.shape[1]))
        cleft[: left.shape[0], : left.shape[1]] = left
        
    if hasattr(right, 'n_outputs'):  # Model
        # This won't happen in our case
        pass
    else:  # ndarray - this is our compound Linear1D & Linear1D
        cright = np.zeros((noutp, right.shape[1]))
        cright[-right.shape[0]:, -right.shape[1]:] = right
        
    return np.hstack([cleft, cright])

# Simulate the separability matrix for Linear1D & Linear1D
linear_compound = np.array([[1, 0], [0, 1]])  # 2x2 diagonal matrix

# Simulate a model with 2 inputs, 2 outputs (like Pix2Sky_TAN)
class MockModel:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

pix2sky = MockModel(2, 2)

# Test the current _cstack implementation
result = _cstack(pix2sky, linear_compound)
print("Current _cstack result:")
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