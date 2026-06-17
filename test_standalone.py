#!/usr/bin/env python
import re
import copy
import tempfile
import os
import numpy as np

# Copied and modified functions from astropy.io.ascii.qdp

def _line_type(line, delimiter=None):
    """Interpret a QDP file line."""
    _decimal_re = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
    _command_re = r"READ [TS]ERR(\s+[0-9]+)+"

    sep = delimiter
    if delimiter is None:
        sep = r"\s+"
    _new_re = rf"NO({sep}NO)+"
    _data_re = rf"({_decimal_re}|NO|[-+]?nan)({sep}({_decimal_re}|NO|[-+]?nan))*)"
    _type_re = rf"^\s*((?P<command>{_command_re})|(?P<new>{_new_re})|(?P<data>{_data_re})?\s*(\!(?P<comment>.*))?\s*$"
    _line_type_re = re.compile(_type_re, re.IGNORECASE)  # Added re.IGNORECASE flag
    line = line.strip()
    if not line:
        return "comment"
    match = _line_type_re.match(line)

    if match is None:
        raise ValueError(f"Unrecognized QDP line: {line}")
    for type_, val in match.groupdict().items():
        if val is None:
            continue
        if type_ == "data":
            return f"data,{len(val.split(sep=delimiter))}"
        else:
            return type_

def _get_type_from_list_of_lines(lines, delimiter=None):
    """Read through the list of QDP file lines and label each line by type."""
    types = [_line_type(line, delimiter=delimiter) for line in lines]
    current_ncol = None
    for type_ in types:
        if type_.startswith("data,"):
            ncol = int(type_[5:])
            if current_ncol is None:
                current_ncol = ncol
            elif ncol != current_ncol:
                raise ValueError("Inconsistent number of columns")

    return types, current_ncol

def _interpret_err_lines(err_specs, ncols, names=None):
    """Give list of column names from the READ SERR and TERR commands."""
    colnames = ["" for i in range(ncols)]
    if err_specs is None:
        serr_cols = terr_cols = []
    else:
        err_specs = copy.deepcopy(err_specs)
        serr_cols = err_specs.pop("serr", [])
        terr_cols = err_specs.pop("terr", [])

    if names is not None:
        all_error_cols = len(serr_cols) + len(terr_cols) * 2
        if all_error_cols + len(names) != ncols:
            raise ValueError("Inconsistent number of input colnames")

    shift = 0
    for i in range(ncols):
        col_num = i + 1 - shift
        if colnames[i] != "":
            continue

        colname_root = f"col{col_num}"

        if names is not None:
            colname_root = names[col_num - 1]

        colnames[i] = f"{colname_root}"
        if col_num in serr_cols:
            colnames[i + 1] = f"{colname_root}_err"
            shift += 1
            continue

        if col_num in terr_cols:
            colnames[i + 1] = f"{colname_root}_perr"
            colnames[i + 2] = f"{colname_root}_nerr"
            shift += 2
            continue

    assert not np.any([c == "" for c in colnames])
    return colnames

class SimpleTable:
    """Simplified table class for testing."""
    def __init__(self, names, rows):
        self.colnames = names
        self.data = {}
        for i, name in enumerate(names):
            self.data[name] = [row[i] for row in rows]
        self.meta = {}
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        if self.colnames:
            return len(self.data[self.colnames[0]])
        return 0

def _get_tables_from_qdp_file(qdp_file, input_colnames=None, delimiter=None):
    """Get all tables from a QDP file."""
    if isinstance(qdp_file, str):
        with open(qdp_file) as fobj:
            lines = [line.strip() for line in fobj.readlines()]
    else:
        lines = qdp_file

    contents, ncol = _get_type_from_list_of_lines(lines, delimiter=delimiter)

    table_list = []
    err_specs = {}
    colnames = None

    comment_text = ""
    initial_comments = ""
    command_lines = ""
    current_rows = None

    for line, datatype in zip(lines, contents):
        line = line.strip().lstrip("!")
        # Is this a comment?
        if datatype == "comment":
            comment_text += line + "\n"
            continue

        if datatype == "command":
            # The first time I find commands, I save whatever comments into
            # The initial comments.
            if command_lines == "":
                initial_comments = comment_text
                comment_text = ""

            command_lines += line + "\n"
            continue

        if datatype.startswith("data"):
            # The first time I find data, I define err_specs
            if err_specs == {} and command_lines != "":
                for cline in command_lines.strip().split("\n"):
                    command = cline.strip().split()
                    # This should never happen, but just in case.
                    if len(command) < 3:
                        continue
                    err_specs[command[1].lower()] = [int(c) for c in command[2:]]
            if colnames is None:
                colnames = _interpret_err_lines(err_specs, ncol, names=input_colnames)

            if current_rows is None:
                current_rows = []

            values = []
            for v in line.split(delimiter):
                if v == "NO":
                    values.append(None)  # Simplified for testing
                else:
                    # Understand if number is int or float
                    try:
                        values.append(int(v))
                    except ValueError:
                        values.append(float(v))
            current_rows.append(values)
            continue

        if datatype == "new":
            # Save table to table_list and reset
            if current_rows is not None:
                new_table = SimpleTable(names=colnames, rows=current_rows)
                new_table.meta["initial_comments"] = initial_comments.strip().split("\n")
                new_table.meta["comments"] = comment_text.strip().split("\n")
                # Reset comments
                comment_text = ""
                table_list.append(new_table)
                current_rows = None
            continue

    # At the very end, if there is still a table being written, let's save
    # it to the table_list
    if current_rows is not None:
        new_table = SimpleTable(names=colnames, rows=current_rows)
        new_table.meta["initial_comments"] = initial_comments.strip().split("\n")
        new_table.meta["comments"] = comment_text.strip().split("\n")
        table_list.append(new_table)

    return table_list

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