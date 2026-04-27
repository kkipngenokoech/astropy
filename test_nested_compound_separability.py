import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    """Test that separability_matrix computes correctly for nested CompoundModels."""
    
    # Create a simple compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability (diagonal)
    cm_sep = separability_matrix(cm)
    expected_cm_sep = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm_sep), f"Simple compound model separability incorrect: got {cm_sep}, expected {expected_cm_sep}"
    
    # Create a more complex compound model without nesting
    complex_model = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    complex_sep = separability_matrix(complex_model)
    expected_complex_sep = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    assert np.array_equal(complex_sep, expected_complex_sep), f"Complex model separability incorrect: got {complex_sep}, expected {expected_complex_sep}"
    
    # Now test the nested compound model - this is where the bug occurs
    nested_model = m.Pix2Sky_TAN() & cm
    nested_sep = separability_matrix(nested_model)
    
    # The expected result should be the same as the non-nested version
    # Pix2Sky_TAN has 2 inputs, 2 outputs; cm has 2 inputs, 2 outputs
    # So the result should be 4x4 with the same pattern as complex_model
    expected_nested_sep = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, False], 
        [False, False, False, True]
    ])
    
    # This assertion will fail on the buggy code because nested CompoundModels
    # are not handled correctly in the _separable function
    assert np.array_equal(nested_sep, expected_nested_sep), f"Nested compound model separability incorrect: got {nested_sep}, expected {expected_nested_sep}"