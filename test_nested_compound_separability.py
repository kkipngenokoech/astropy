import numpy as np
from astropy.modeling.models import Linear1D
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create the nested compound model as described in the issue
    # First create simple compound models that work correctly
    m1 = Linear1D(1) & Linear1D(2)
    m2 = Linear1D(3) & Linear1D(4)
    
    # Verify the simple compound models have correct separability (diagonal)
    sep1 = separability_matrix(m1)
    sep2 = separability_matrix(m2)
    expected_diagonal = np.array([[True, False], [False, True]])
    assert np.array_equal(sep1, expected_diagonal)
    assert np.array_equal(sep2, expected_diagonal)
    
    # Now create the nested compound model
    nested_model = m1 | m2
    
    # The separability matrix should still be diagonal since the linear models
    # are independent, but the current implementation incorrectly shows dependencies
    nested_sep = separability_matrix(nested_model)
    
    # This assertion should pass (diagonal matrix) but will fail due to the bug
    # The bug causes the matrix to show false dependencies between inputs/outputs
    assert np.array_equal(nested_sep, expected_diagonal), f"Expected diagonal matrix but got:\n{nested_sep}"