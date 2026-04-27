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
    
    # The nested model should maintain separability:
    # - Pix2Sky_TAN takes 2 inputs, produces 2 outputs (indices 0,1)
    # - First Linear1D takes 1 input, produces 1 output (index 2)
    # - Second Linear1D takes 1 input, produces 1 output (index 3)
    # So outputs 2 and 3 should be independent of each other and of outputs 0,1
    nested_sep = separability_matrix(nested)
    expected_nested = np.array([
        [True,  True,  False, False],  # output 0 depends on inputs 0,1
        [True,  True,  False, False],  # output 1 depends on inputs 0,1  
        [False, False, True,  False],  # output 2 depends only on input 2
        [False, False, False, True ]   # output 3 depends only on input 3
    ])
    
    # This assertion will FAIL on the buggy code because it incorrectly
    # shows dependencies between the Linear1D outputs
    assert np.array_equal(nested_sep, expected_nested), f"Expected {expected_nested}, got {nested_sep}"

if __name__ == "__main__":
    test_issue_reproduction()
    print("All tests passed!")