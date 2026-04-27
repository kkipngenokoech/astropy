import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    """Test that separability_matrix computes correctly for nested CompoundModels."""
    
    # First, create the simple compound model that works correctly
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability (diagonal)
    simple_result = separability_matrix(cm)
    expected_simple = np.array([[ True, False],
                               [False,  True]])
    assert np.array_equal(simple_result, expected_simple), f"Simple compound model failed: got {simple_result}, expected {expected_simple}"
    
    # Test the more complex model that should work
    complex_model = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    complex_result = separability_matrix(complex_model)
    expected_complex = np.array([[ True,  True, False, False],
                                [False, False,  True, False],
                                [False, False, False,  True]])
    assert np.array_equal(complex_result, expected_complex), f"Complex model failed: got {complex_result}, expected {expected_complex}"
    
    # Now test the nested compound model that currently fails
    nested_model = m.Pix2Sky_TAN() & cm
    nested_result = separability_matrix(nested_model)
    
    # The expected result should be the same as the complex model above
    # because Pix2Sky_TAN() has 2 inputs/2 outputs, and cm has 2 inputs/2 outputs
    # So the combined model should have 4 inputs and 4 outputs with proper separability
    expected_nested = np.array([[ True,  True, False, False],
                               [False, False,  True, False], 
                               [False, False, False,  True]])
    
    # This assertion should fail on the current buggy code
    assert np.array_equal(nested_result, expected_nested), f"Nested compound model failed: got {nested_result}, expected {expected_nested}"