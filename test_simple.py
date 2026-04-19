#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from astropy.io.ascii.qdp import _line_type

# Test the current behavior
try:
    result = _line_type("READ SERR 1 2")
    print(f"Uppercase command works: {result}")
except Exception as e:
    print(f"Uppercase command failed: {e}")

try:
    result = _line_type("read serr 1 2")
    print(f"Lowercase command works: {result}")
except Exception as e:
    print(f"Lowercase command failed: {e}")