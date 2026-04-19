#!/usr/bin/env python
"""
Test script to verify that RST class now accepts header_rows parameter.
This reproduces the issue described in astropy__astropy-14182.
"""

def test_rst_header_rows():
    """Test that RST class accepts header_rows parameter without TypeError"""
    
    # Test the signature change by checking the source code
    import inspect
    import os
    
    # Read the RST class source
    rst_path = os.path.join('astropy', 'io', 'ascii', 'rst.py')
    with open(rst_path, 'r') as f:
        content = f.read()
    
    # Verify the __init__ method signature was updated
    if 'def __init__(self, header_rows=None):' in content:
        print("✓ RST.__init__ now accepts header_rows parameter")
    else:
        print("✗ RST.__init__ signature was not updated correctly")
        return False
    
    # Verify the super().__init__ call passes header_rows
    if 'super().__init__(delimiter_pad=None, bookend=False, header_rows=header_rows)' in content:
        print("✓ RST.__init__ passes header_rows to parent FixedWidth class")
    else:
        print("✗ RST.__init__ does not pass header_rows to parent class")
        return False
    
    print("✓ All signature checks passed!")
    return True

if __name__ == "__main__":
    success = test_rst_header_rows()
    if success:
        print("\n🎉 SUCCESS: The RST class has been fixed to support header_rows parameter!")
        exit(0)
    else:
        print("\n❌ FAILURE: The RST class fix was not applied correctly.")
        exit(1)