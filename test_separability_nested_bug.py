import numpy as np
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

def test_issue_reproduction():
    """Test that separability_matrix computes correctly for nested CompoundModels."""
    # Create a simple compound model
    cm = m.Linear1D(10) & m.Linear1D(5)
    
    # Verify the simple compound model has correct separability (diagonal)
    cm_sep = separability_matrix(cm)
    expected_cm = np.array([[True, False], [False, True]])
    assert np.array_equal(cm_sep, expected_cm), f"Simple compound model separability incorrect: {cm_sep}"
    
    # Create a more complex compound model
    complex_model = m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5)
    complex_sep = separability_matrix(complex_model)
    expected_complex = np.array([
        [True, True, False, False],
        [True, True, False, False], 
        [False, False, True, False],
        [False, False, False, True]
    ])
    assert np.array_equal(complex_sep, expected_complex), f"Complex compound model separability incorrect: {complex_sep}"
    
    # Now test the nested compound model - this should have the same separability as the complex model
    nested_model = m.Pix2Sky_TAN() & cm
    nested_sep = separability_matrix(nested_model)
    
    # The nested model should have the same separability as the complex model
    # since cm = m.Linear1D(10) & m.Linear1D(5) should be equivalent to having those models directly
    expected_nested = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, True, False], 
        [False, False, False, True]
    ])
    
    # This assertion should fail on the buggy code
    assert np.array_equal(nested_sep, expected_nested), f"Nested compound model separability incorrect: {nested_sep}"

if __name__ == "__main__":
    test_issue_reproduction()
    print("All tests passed!")
