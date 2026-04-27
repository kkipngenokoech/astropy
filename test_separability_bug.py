import numpy as np
from astropy.modeling.models import Linear1D
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create nested compound models as described in the issue
    # First level: simple compound models that work correctly
    m1 = Linear1D(1) & Linear1D(2)
    m2 = Linear1D(3) & Linear1D(4)
    
    # Verify the simple compound models have correct separability (diagonal)
    sep1 = separability_matrix(m1)
    sep2 = separability_matrix(m2)
    expected_diagonal = np.array([[True, False], [False, True]])
    assert np.array_equal(sep1, expected_diagonal)
    assert np.array_equal(sep2, expected_diagonal)
    
    # Now nest these compound models - this should maintain separability
    # but currently fails due to the bug
    nested_model = m1 | m2
    nested_sep = separability_matrix(nested_model)
    
    # The nested model should still be separable (diagonal matrix)
    # because the linear models are independent
    # But due to the bug, this will fail
    assert np.array_equal(nested_sep, expected_diagonal), f"Expected diagonal separability matrix, got {nested_sep}"