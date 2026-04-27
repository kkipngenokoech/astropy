import io
from astropy.io import ascii
from astropy.table import Table

def test_issue_reproduction():
    # Create a simple table with column names (headers)
    t = Table()
    t['Col1'] = [1, 2]
    t['Col2'] = [2.3, 4.5]
    t['Col3'] = ['Hello', 'Worlds']
    
    # Try to write it in RST format
    output = io.StringIO()
    ascii.write(t, output, format='rst')
    result = output.getvalue()
    
    # The output should have proper RST header format with column names
    # and separator lines made of '=' characters
    lines = result.strip().split('\n')
    
    # Check that we have the expected RST structure:
    # Line 0: separator line with '=' chars
    # Line 1: header line with column names
    # Line 2: separator line with '=' chars  
    # Line 3+: data rows
    # Last line: separator line with '=' chars
    
    # This should fail because current implementation doesn't properly
    # format headers in RST output
    assert 'Col1' in lines[1]  # Header row should contain column names
    assert 'Col2' in lines[1]
    assert 'Col3' in lines[1]
    assert lines[0].startswith('====')  # First line should be separator
    assert lines[2].startswith('====')  # Third line should be separator after header