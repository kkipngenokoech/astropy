import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create a compound model of two Linear1D models
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability
    cm_sep = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm), f"Expected {expected_cm}, got {cm_sep}"
    
    # Create a more complex model without nesting
    simple_complex = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    simple_sep = separability_matrix(simple_complex)
    expected_simple = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    assert np.array_equal(simple_sep, expected_simple), f"Expected {expected_simple}, got {simple_sep}"
    
    # Now test the nested compound model - this should have the same separability
    nested = m.Pix2Sky_TAN() & cm
    nested_sep = separability_matrix(nested)
    
    # The nested model should have the same separability as the simple complex model
    # But the current implementation incorrectly shows dependencies
    assert np.array_equal(nested_sep, expected_simple), f"Nested model separability incorrect: expected {expected_simple}, got {nested_sep}"