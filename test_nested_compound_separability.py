import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    """Test that nested CompoundModels preserve separability correctly."""
    
    # Create the inner compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the inner compound model has correct separability (diagonal)
    cm_sep = separability_matrix(cm)
    expected_cm_sep = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm_sep), f"Inner compound model separability incorrect: {cm_sep}"
    
    # Create the nested compound model
    nested_model = m.Pix2Sky_TAN() & cm
    
    # The nested model should preserve separability:
    # - Pix2Sky_TAN has 2 inputs, 2 outputs (both coupled)
    # - Linear1D(10) has 1 input, 1 output (independent)
    # - Linear1D(5) has 1 input, 1 output (independent)
    # Total: 4 inputs, 4 outputs
    # Expected separability matrix should be:
    # [[True, True, False, False],   # Pix2Sky_TAN output 0 depends on inputs 0,1
    #  [True, True, False, False],   # Pix2Sky_TAN output 1 depends on inputs 0,1  
    #  [False, False, True, False],  # Linear1D(10) output depends only on input 2
    #  [False, False, False, True]]  # Linear1D(5) output depends only on input 3
    
    nested_sep = separability_matrix(nested_model)
    expected_nested_sep = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    
    # This assertion should FAIL on the current buggy code
    assert np.array_equal(nested_sep, expected_nested_sep), (
        f"Nested compound model separability incorrect:\n"
        f"Got:\n{nested_sep}\n"
        f"Expected:\n{expected_nested_sep}"
    )