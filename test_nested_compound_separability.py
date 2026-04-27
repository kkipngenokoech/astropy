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
    
    # Create a more complex model by stacking independent models
    complex_model = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    complex_sep = separability_matrix(complex_model)
    expected_complex = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    assert np.array_equal(complex_sep, expected_complex), f"Complex model separability incorrect: {complex_sep}"
    
    # Now test the nested compound model - this should have the same separability as above
    # since we're just replacing two Linear1D models with their compound equivalent
    nested_model = m.Pix2Sky_TAN() & cm
    nested_sep = separability_matrix(nested_model)
    
    # The nested model should have the same separability as the complex model
    # because cm is separable and we're just stacking it with Pix2Sky_TAN
    assert np.array_equal(nested_sep, expected_complex), f"Nested compound model separability incorrect: {nested_sep}"