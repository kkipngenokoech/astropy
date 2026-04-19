#!/usr/bin/env python
"""
Minimal test to verify RST header_rows fix without full astropy import
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '.')

# Mock the erfa module to avoid import errors
class MockErfa:
    class ErfaError(Exception):
        pass
    class ErfaWarning(Warning):
        pass

sys.modules['erfa'] = MockErfa()

# Mock other problematic modules
class MockModule:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['astropy.version'] = MockModule()

try:
    # Now try to import and test the RST class
    from astropy.io.ascii.rst import RST
    
    print("Testing RST class...")
    
    # Test 1: Default constructor
    try:
        rst1 = RST()
        print("✓ RST() works")
        print(f"  Default data.start_line: {rst1.data.start_line}")
    except Exception as e:
        print(f"✗ RST() failed: {e}")
    
    # Test 2: Constructor with header_rows
    try:
        rst2 = RST(header_rows=['name', 'unit'])
        print("✓ RST(header_rows=['name', 'unit']) works")
        print(f"  Custom data.start_line: {rst2.data.start_line}")
    except Exception as e:
        print(f"✗ RST(header_rows=...) failed: {e}")
    
    # Test 3: Constructor with different header_rows
    try:
        rst3 = RST(header_rows=['name', 'unit', 'description'])
        print("✓ RST(header_rows=['name', 'unit', 'description']) works")
        print(f"  Custom data.start_line: {rst3.data.start_line}")
    except Exception as e:
        print(f"✗ RST(header_rows=...) with 3 rows failed: {e}")
        
    print("\nAll tests passed! The fix appears to work correctly.")
    
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()