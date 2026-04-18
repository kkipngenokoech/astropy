import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create a simple compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability (diagonal)
    cm_sep = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm), f"Simple compound model separability incorrect: {cm_sep}"
    
    # Create a nested compound model
    nested = m.Pix2Sky_TAN() & cm
    
    # The nested model should have separable outputs:
    # - Pix2Sky_TAN takes 2 inputs, produces 2 outputs (indices 0,1)
    # - First Linear1D takes 1 input, produces 1 output (index 2) 
    # - Second Linear1D takes 1 input, produces 1 output (index 3)
    # So the separability matrix should be:
    # [[True, True, False, False],   # output 0 depends on inputs 0,1 (Pix2Sky_TAN)
    #  [True, True, False, False],   # output 1 depends on inputs 0,1 (Pix2Sky_TAN)
    #  [False, False, True, False],  # output 2 depends on input 2 (first Linear1D)
    #  [False, False, False, True]]  # output 3 depends on input 3 (second Linear1D)
    
    nested_sep = separability_matrix(nested)
    expected_nested = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    
    # This assertion should fail on the current buggy code
    assert np.array_equal(nested_sep, expected_nested), f"Nested compound model separability incorrect: {nested_sep}"