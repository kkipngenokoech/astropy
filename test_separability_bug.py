import numpy as np
from astropy.modeling.models import Linear1D
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    # Create nested compound models as described in the issue
    # First level: simple compound models that work correctly
    m1 = Linear1D(slope=1, intercept=2) & Linear1D(slope=2, intercept=3)
    m2 = Linear1D(slope=1, intercept=2) & Linear1D(slope=2, intercept=3)
    
    # Second level: nest the compound models
    nested_model = m1 | m2
    
    # Get the separability matrix
    sep_matrix = separability_matrix(nested_model)
    
    # For this nested model, the separability matrix should be diagonal
    # (each output should depend only on its corresponding input)
    expected_matrix = np.array([[True, False], [False, True]])
    
    # This assertion should pass but will fail due to the bug
    # The bug causes the matrix to show all True values instead of diagonal
    assert np.array_equal(sep_matrix, expected_matrix), f"Expected diagonal matrix {expected_matrix}, but got {sep_matrix}"