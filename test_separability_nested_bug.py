import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create a compound model of two Linear1D models
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability (diagonal)
    cm_sep = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm), f"Expected {expected_cm}, got {cm_sep}"
    
    # Create a nested compound model
    nested = m.Pix2Sky_TAN() & cm
    
    # The nested model should have separable outputs:
    # - Outputs 0,1 (from Pix2Sky_TAN) should depend only on inputs 0,1
    # - Output 2 (from first Linear1D) should depend only on input 2  
    # - Output 3 (from second Linear1D) should depend only on input 3
    nested_sep = separability_matrix(nested)
    expected_nested = np.array([
        [True, True, False, False],   # Pix2Sky_TAN output 0 depends on inputs 0,1
        [True, True, False, False],   # Pix2Sky_TAN output 1 depends on inputs 0,1  
        [False, False, True, False],  # Linear1D(10) output depends only on input 2
        [False, False, False, True]   # Linear1D(5) output depends only on input 3
    ])
    
    # This assertion will fail on the buggy code because it incorrectly
    # shows dependencies between the Linear1D outputs and Pix2Sky_TAN inputs
    assert np.array_equal(nested_sep, expected_nested), f"Expected {expected_nested}, got {nested_sep}"