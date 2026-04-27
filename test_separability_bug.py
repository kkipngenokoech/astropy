import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create a compound model as described in the issue
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model works correctly
    cm_matrix = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_matrix, expected_cm), f"Expected {expected_cm}, got {cm_matrix}"
    
    # Create nested compound model that demonstrates the bug
    nested_model = m.Pix2Sky_TAN() & cm
    
    # The separability matrix should be diagonal - each output should depend on its corresponding inputs
    # Pix2Sky_TAN has 2 inputs, 2 outputs
    # cm has 2 inputs, 2 outputs  
    # So nested_model should have 4 inputs, 4 outputs with diagonal separability
    nested_matrix = separability_matrix(nested_model)
    expected_nested = np.array([
        [True, True, False, False],   # First Pix2Sky_TAN output depends on first 2 inputs
        [True, True, False, False],   # Second Pix2Sky_TAN output depends on first 2 inputs  
        [False, False, True, False],  # First Linear1D output depends on 3rd input
        [False, False, False, True]   # Second Linear1D output depends on 4th input
    ])
    
    # This assertion will fail on the current buggy code
    assert np.array_equal(nested_matrix, expected_nested), f"Expected {expected_nested}, got {nested_matrix}"