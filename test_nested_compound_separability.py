import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    """Test that nested CompoundModels compute separability correctly."""
    
    # Create a simple compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability
    cm_sep = separability_matrix(cm)
    expected_cm_sep = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm_sep), f"Simple compound model separability incorrect: {cm_sep}"
    
    # Create a flattened version (should work correctly)
    flattened = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    flattened_sep = separability_matrix(flattened)
    
    # Create nested version (this is the bug case)
    nested = m.Pix2Sky_TAN() & cm
    nested_sep = separability_matrix(nested)
    
    # The nested version should have the same separability as the flattened version
    # Both should have:
    # - First 2 outputs depend on first 2 inputs (Pix2Sky_TAN)
    # - Third output depends only on third input (first Linear1D)
    # - Fourth output depends only on fourth input (second Linear1D)
    expected_sep = np.array([
        [True, True, False, False],   # First output of Pix2Sky_TAN
        [True, True, False, False],   # Second output of Pix2Sky_TAN  
        [False, False, True, False],  # Output of first Linear1D
        [False, False, False, True]   # Output of second Linear1D
    ])
    
    # Verify flattened version works (this should pass)
    assert np.array_equal(flattened_sep, expected_sep), f"Flattened separability incorrect: {flattened_sep}"
    
    # This assertion should fail due to the bug - nested compound models don't compute separability correctly
    assert np.array_equal(nested_sep, expected_sep), f"Nested compound model separability incorrect: {nested_sep}, expected: {expected_sep}"