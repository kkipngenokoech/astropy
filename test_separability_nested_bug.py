import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create a simple compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # This should work correctly (baseline)
    simple_result = separability_matrix(cm)
    expected_simple = np.array([[True, False], [False, True]])
    assert np.array_equal(simple_result, expected_simple)
    
    # Create a more complex model without nesting
    complex_unnested = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    complex_result = separability_matrix(complex_unnested)
    expected_complex = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    assert np.array_equal(complex_result, expected_complex)
    
    # Now test the nested case - this should give the same result as the unnested case
    # but currently fails due to the bug
    nested = m.Pix2Sky_TAN() & cm
    nested_result = separability_matrix(nested)
    
    # The nested result should be the same as the unnested result
    # because we're just concatenating the same models
    assert np.array_equal(nested_result, expected_complex), f"Expected {expected_complex}, got {nested_result}"