import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create a compound model as described in the issue
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the basic compound model works correctly
    cm_matrix = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_matrix, expected_cm), f"Expected {expected_cm}, got {cm_matrix}"
    
    # Create the nested compound model that exhibits the bug
    nested_model = m.Pix2Sky_TAN() & cm
    
    # Get the separability matrix for the nested model
    nested_matrix = separability_matrix(nested_model)
    
    # The expected matrix should show that:
    # - First 2 outputs (from Pix2Sky_TAN) depend on first 2 inputs
    # - Third output (from first Linear1D) depends only on third input  
    # - Fourth output (from second Linear1D) depends only on fourth input
    expected_nested = np.array([
        [True, True, False, False],   # First TAN output
        [True, True, False, False],   # Second TAN output  
        [False, False, True, False],  # First Linear1D output
        [False, False, False, True]   # Second Linear1D output
    ])
    
    # This assertion should fail with the current buggy code
    assert np.array_equal(nested_matrix, expected_nested), f"Expected {expected_nested}, got {nested_matrix}"

if __name__ == "__main__":
    test_issue_reproduction()
    print("Test passed!")