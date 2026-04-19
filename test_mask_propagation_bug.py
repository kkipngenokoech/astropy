import numpy as np
from astropy.nddata import NDDataRef

def test_issue_reproduction():
    # Create two NDDataRef objects where one has a mask and one doesn't
    data1 = np.array([1, 2, 3])
    mask1 = np.array([True, False, False])  # First operand has a mask
    ndd1 = NDDataRef(data1, mask=mask1)
    
    data2 = np.array([4, 5, 6])
    ndd2 = NDDataRef(data2)  # Second operand has no mask (mask=None)
    
    # This should work but fails in v5.3 when handle_mask=np.bitwise_or
    # The error occurs because bitwise_or tries to operate on mask1 and None
    result = ndd1.add(ndd2, handle_mask=np.bitwise_or)
    
    # Test the reverse case too
    result2 = ndd2.add(ndd1, handle_mask=np.bitwise_or)
    
    print("Test passed!")

if __name__ == "__main__":
    test_issue_reproduction()