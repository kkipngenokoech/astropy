import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    """Test that nested CompoundModels compute separability correctly."""
    # Create a simple compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability
    cm_sep = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm), f"Expected {expected_cm}, got {cm_sep}"
    
    # Create a nested compound model
    nested = m.Pix2Sky_TAN() & cm
    
    # The nested model should have separable outputs:
    # - Pix2Sky_TAN takes 2 inputs, produces 2 outputs (indices 0,1)
    # - First Linear1D takes 1 input, produces 1 output (index 2) 
    # - Second Linear1D takes 1 input, produces 1 output (index 3)
    # So outputs should be independent: [0,1] depend on inputs [0,1], 
    # output [2] depends on input [2], output [3] depends on input [3]
    nested_sep = separability_matrix(nested)
    expected_nested = np.array([
        [True, True, False, False],   # output 0 depends on inputs 0,1
        [True, True, False, False],   # output 1 depends on inputs 0,1  
        [False, False, True, False],  # output 2 depends on input 2
        [False, False, False, True]   # output 3 depends on input 3
    ])
    
    # This assertion will fail on the buggy code
    assert np.array_equal(nested_sep, expected_nested), f"Expected {expected_nested}, got {nested_sep}"