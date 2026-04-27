#!/usr/bin/env python3

import numpy as np

# Copy the relevant functions from separable.py to test them directly

def _compute_n_outputs(left, right):
    """
    Compute the number of outputs of two models.
    """
    if hasattr(left, 'n_outputs'):
        lnout = left.n_outputs
    else:
        lnout = left.shape[0]
    if hasattr(right, 'n_outputs'):
        rnout = right.n_outputs
    else:
        rnout = right.shape[0]
    noutp = lnout + rnout
    return noutp

def _cstack_old(left, right):
    """
    Original buggy version of _cstack function.
    """
    noutp = _compute_n_outputs(left, right)

    if hasattr(left, 'n_outputs'):  # It's a model
        # Mock _coord_matrix behavior for left
        cleft = np.zeros((noutp, left.n_inputs))
        if not left.separable:
            cleft[:left.n_outputs, :left.n_inputs] = 1
        else:
            for i in range(left.n_inputs):
                cleft[i, i] = 1
    else:
        cleft = np.zeros((noutp, left.shape[1]))
        cleft[: left.shape[0], : left.shape[1]] = left
        
    if hasattr(right, 'n_outputs'):  # It's a model
        # Mock _coord_matrix behavior for right
        cright = np.zeros((noutp, right.n_inputs))
        if not right.separable:
            cright[-right.n_outputs:, -right.n_inputs:] = 1
        else:
            for i in range(right.n_inputs):
                cright[-(right.n_outputs-i), -(right.n_inputs-i)] = 1
    else:
        cright = np.zeros((noutp, right.shape[1]))
        cright[-right.shape[0]:, -right.shape[1]:] = 1  # BUG: should be 'right'

    return np.hstack([cleft, cright])

def _cstack_fixed(left, right):
    """
    Fixed version of _cstack function.
    """
    noutp = _compute_n_outputs(left, right)

    if hasattr(left, 'n_outputs'):  # It's a model
        # Mock _coord_matrix behavior for left
        cleft = np.zeros((noutp, left.n_inputs))
        if not left.separable:
            cleft[:left.n_outputs, :left.n_inputs] = 1
        else:
            for i in range(left.n_inputs):
                cleft[i, i] = 1
    else:
        cleft = np.zeros((noutp, left.shape[1]))
        cleft[: left.shape[0], : left.shape[1]] = left
        
    if hasattr(right, 'n_outputs'):  # It's a model
        # Mock _coord_matrix behavior for right
        cright = np.zeros((noutp, right.n_inputs))
        if not right.separable:
            cright[-right.n_outputs:, -right.n_inputs:] = 1
        else:
            for i in range(right.n_inputs):
                cright[-(right.n_outputs-i), -(right.n_inputs-i)] = 1
    else:
        cright = np.zeros((noutp, right.shape[1]))
        cright[-right.shape[0]:, -right.shape[1]:] = right  # FIX: use 'right' instead of 1

    return np.hstack([cleft, cright])

# Mock model class
class MockModel:
    def __init__(self, n_inputs, n_outputs, separable=True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.separable = separable

def test_fix():
    print("Testing the fix for _cstack function...")
    
    # Create a mock model (Pix2Sky_TAN: 2 inputs, 2 outputs, not separable)
    pix2sky_matrix = np.array([[1, 1], [1, 1]], dtype=float)  # Not separable
    
    # Create a separability matrix for compound model (Linear1D & Linear1D)
    compound_matrix = np.array([[1, 0], [0, 1]], dtype=float)  # Separable
    
    print("Left matrix (Pix2Sky_TAN):")
    print(pix2sky_matrix)
    print("Right matrix (Linear1D & Linear1D):")
    print(compound_matrix)
    
    # Test old (buggy) version
    print("\nOld (buggy) result:")
    old_result = _cstack_old(pix2sky_matrix, compound_matrix)
    print(old_result)
    
    # Test new (fixed) version
    print("\nNew (fixed) result:")
    new_result = _cstack_fixed(pix2sky_matrix, compound_matrix)
    print(new_result)
    
    # Expected result
    expected = np.array([
        [1, 1, 0, 0],  # Pix2Sky_TAN output 0 depends on inputs 0,1
        [1, 1, 0, 0],  # Pix2Sky_TAN output 1 depends on inputs 0,1
        [0, 0, 1, 0],  # Linear1D output 0 depends only on input 2
        [0, 0, 0, 1]   # Linear1D output 1 depends only on input 3
    ], dtype=float)
    
    print("\nExpected result:")
    print(expected)
    
    print("\nComparison:")
    print(f"Old result matches expected: {np.array_equal(old_result, expected)}")
    print(f"New result matches expected: {np.array_equal(new_result, expected)}")
    
    return np.array_equal(new_result, expected)

if __name__ == "__main__":
    success = test_fix()
    if success:
        print("\n✅ Fix is working correctly!")
    else:
        print("\n❌ Fix is not working!")
    
    print("\nThe bug was in line:")
    print("  cright[-right.shape[0]:, -right.shape[1]:] = 1")
    print("Should be:")
    print("  cright[-right.shape[0]:, -right.shape[1]:] = right")