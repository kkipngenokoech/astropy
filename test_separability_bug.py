import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create a simple compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # This should work correctly - diagonal separability matrix
    cm_sep = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm), f"Simple compound model failed: got {cm_sep}, expected {expected_cm}"
    
    # Create a more complex model that should also be separable
    complex_model = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    complex_sep = separability_matrix(complex_model)
    expected_complex = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    assert np.array_equal(complex_sep, expected_complex), f"Complex model failed: got {complex_sep}, expected {expected_complex}"
    
    # Now test the nested compound model - this should fail with current code
    nested_model = m.Pix2Sky_TAN() & cm
    nested_sep = separability_matrix(nested_model)
    expected_nested = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, False], 
        [False, False, False, True]
    ])
    
    # This assertion should fail with the current buggy code
    assert np.array_equal(nested_sep, expected_nested), f"Nested compound model failed: got {nested_sep}, expected {expected_nested}"