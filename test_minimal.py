#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# Create minimal mock modules to avoid import dependencies
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

# Mock the problematic modules
sys.modules['astropy.version'] = MockModule()
sys.modules['erfa'] = MockModule()
sys.modules['astropy.utils'] = MockModule()
sys.modules['astropy.utils.exceptions'] = MockModule()
sys.modules['astropy.utils.decorators'] = MockModule()
sys.modules['astropy.utils.introspection'] = MockModule()
sys.modules['astropy.utils.codegen'] = MockModule()
sys.modules['astropy.config'] = MockModule()
sys.modules['astropy.config.configuration'] = MockModule()

# Now let's test the RST class by importing it directly
try:
    # Import the core modules we need
    from astropy.io.ascii.core import DefaultSplitter
    from astropy.io.ascii.fixedwidth import (
        FixedWidth,
        FixedWidthData,
        FixedWidthHeader,
        FixedWidthTwoLineDataSplitter,
    )
    
    # Import the RST class
    from astropy.io.ascii.rst import RST
    
    print("Successfully imported RST class")
    
    # Test the constructor with header_rows parameter
    try:
        rst = RST(header_rows=2)
        print("SUCCESS: RST(header_rows=2) works!")
    except TypeError as e:
        if "header_rows" in str(e):
            print(f"FAIL: RST still doesn't accept header_rows: {e}")
        else:
            print(f"FAIL: Different TypeError: {e}")
    except Exception as e:
        print(f"FAIL: Unexpected exception: {e}")
    
    # Test default constructor
    try:
        rst_default = RST()
        print("SUCCESS: RST() default constructor works!")
    except Exception as e:
        print(f"FAIL: Default constructor failed: {e}")
        
except Exception as e:
    print(f"Import or test failed: {e}")
    import traceback
    traceback.print_exc()