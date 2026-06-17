#!/usr/bin/env python
import sys
import os
import tempfile

# Add the current directory to the path so we can import the modified module
sys.path.insert(0, os.path.abspath('.'))

# Import the functions we need
from astropy.io.ascii.qdp import _get_tables_from_qdp_file

def test_case_insensitive_commands():
    """Test that QDP files with lowercase commands work."""
    
    # Test data with lowercase command
    qdp_content = "read serr 1 2\n1 0.5 1 0.5\n"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should work now with our fix
        tables = _get_tables_from_qdp_file(temp_filename)
        
        # Verify we got a table
        assert len(tables) == 1
        table = tables[0]
        
        # Verify the table structure
        print(f"Table columns: {table.colnames}")
        print(f"Table data: {table}")
        
        # Check that we have the expected columns (2 data + 2 error columns)
        assert len(table.colnames) == 4
        assert 'col1' in table.colnames
        assert 'col1_err' in table.colnames
        assert 'col2' in table.colnames
        assert 'col2_err' in table.colnames
        
        # Check the data values
        assert len(table) == 1
        assert table['col1'][0] == 1
        assert table['col1_err'][0] == 0.5
        assert table['col2'][0] == 1
        assert table['col2_err'][0] == 0.5
        
        print("✓ Lowercase command test passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
    finally:
        # Clean up
        os.unlink(temp_filename)

def test_mixed_case_commands():
    """Test that QDP files with mixed case commands work."""
    
    # Test data with mixed case command
    qdp_content = "Read Serr 1 2\n1 0.5 1 0.5\n"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should work now with our fix
        tables = _get_tables_from_qdp_file(temp_filename)
        
        # Verify we got a table
        assert len(tables) == 1
        table = tables[0]
        
        # Check that we have the expected columns (2 data + 2 error columns)
        assert len(table.colnames) == 4
        
        print("✓ Mixed case command test passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
    finally:
        # Clean up
        os.unlink(temp_filename)

def test_uppercase_still_works():
    """Test that uppercase commands still work (regression test)."""
    
    # Test data with uppercase command (original format)
    qdp_content = "READ SERR 1 2\n1 0.5 1 0.5\n"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qdp', delete=False) as f:
        f.write(qdp_content)
        temp_filename = f.name
    
    try:
        # This should still work
        tables = _get_tables_from_qdp_file(temp_filename)
        
        # Verify we got a table
        assert len(tables) == 1
        table = tables[0]
        
        # Check that we have the expected columns (2 data + 2 error columns)
        assert len(table.colnames) == 4
        
        print("✓ Uppercase command test passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise
    finally:
        # Clean up
        os.unlink(temp_filename)

if __name__ == "__main__":
    test_case_insensitive_commands()
    test_mixed_case_commands()
    test_uppercase_still_works()
    print("All tests passed!")