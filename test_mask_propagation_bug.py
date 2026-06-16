import numpy as np
from astropy.nddata import NDDataRef
import pytest

def test_issue_reproduction():
    """Test that mask propagation fails when one operand has no mask and handle_mask=np.bitwise_or."""
    # Create two NDDataRef objects - one with mask, one without
    data1 = np.array([1, 2, 3])
    mask1 = np.array([True, False, False])
    ndd1 = NDDataRef(data1, mask=mask1)
    
    data2 = np.array([4, 5, 6])
    ndd2 = NDDataRef(data2)  # No mask (mask=None)
    
    # This should fail with TypeError: unsupported operand type(s) for |: 'int' and 'NoneType'
    # when handle_mask=np.bitwise_or is used
    with pytest.raises(TypeError, match="unsupported operand type\(s\) for \|: 'int' and 'NoneType'"):
        result = ndd1.add(ndd2, handle_mask=np.bitwise_or)
    
    # Also test the reverse case (operand without mask first)
    with pytest.raises(TypeError, match="unsupported operand type\(s\) for \|: 'int' and 'NoneType'"):
        result = ndd2.add(ndd1, handle_mask=np.bitwise_or)
    
    # Test that it should work with default handle_mask (np.logical_or)
    result_default = ndd1.add(ndd2)  # Should work fine
    expected_mask = np.array([True, False, False])  # Should copy mask from ndd1
    np.testing.assert_array_equal(result_default.mask, expected_mask)
    
    # Test that it should work with 'first_found' handle_mask
    result_ff = ndd1.add(ndd2, handle_mask='first_found')
    np.testing.assert_array_equal(result_ff.mask, expected_mask)