#!/usr/bin/env python3

# Simple test to verify RST class accepts header_rows parameter
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the astropy.version module to avoid import errors
class MockVersion:
    def __init__(self):
        pass

sys.modules['astropy.version'] = MockVersion()

# Mock the erfa module to avoid import errors
class MockErfa:
    class ErfaError(Exception):
        pass
    class ErfaWarning(Warning):
        pass

sys.modules['erfa'] = MockErfa()

# Now try to import and test the RST class
try:
    # Import the necessary modules directly
    from astropy.io.ascii.rst import RST
    
    # Test 1: RST should accept header_rows parameter without raising TypeError
    try:
        rst = RST(header_rows=2)
        print("SUCCESS: RST(header_rows=2) works without TypeError")
    except TypeError as e:
        print(f"FAIL: RST(header_rows=2) raised TypeError: {e}")
        sys.exit(1)
    
    # Test 2: RST should work with default parameters
    try:
        rst_default = RST()
        print("SUCCESS: RST() works with default parameters")
    except Exception as e:
        print(f"FAIL: RST() raised exception: {e}")
        sys.exit(1)
    
    # Test 3: RST should work with header_rows=None
    try:
        rst_none = RST(header_rows=None)
        print("SUCCESS: RST(header_rows=None) works")
    except Exception as e:
        print(f"FAIL: RST(header_rows=None) raised exception: {e}")
        sys.exit(1)
        
    print("All tests passed!")
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)