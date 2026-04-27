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
    # - Pix2Sky_TAN takes 2 inputs, produces 2 outputs (both depend on both inputs)
    # - cm takes 2 inputs, produces 2 outputs (each output depends on one input)
    # So the combined model should be:
    # Output 0 depends on inputs 0,1 (from Pix2Sky_TAN)
    # Output 1 depends on inputs 0,1 (from Pix2Sky_TAN) 
    # Output 2 depends on input 2 (from first Linear1D in cm)
    # Output 3 depends on input 3 (from second Linear1D in cm)
    nested_sep = separability_matrix(nested)
    expected_nested = np.array([
        [True, True, False, False],  # Output 0 depends on inputs 0,1
        [True, True, False, False],  # Output 1 depends on inputs 0,1
        [False, False, True, False], # Output 2 depends on input 2
        [False, False, False, True]  # Output 3 depends on input 3
    ])
    
    # This assertion will fail on the current buggy code
    assert np.array_equal(nested_sep, expected_nested), f"Expected {expected_nested}, got {nested_sep}"