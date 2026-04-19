#!/usr/bin/env python
import re

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

# Test the fixed behavior
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

try:
    result = _line_type("Read Serr 1 2")
    print(f"Mixed case command works: {result}")
except Exception as e:
    print(f"Mixed case command failed: {e}")

try:
    result = _line_type("READ TERR 1 2")
    print(f"Uppercase TERR command works: {result}")
except Exception as e:
    print(f"Uppercase TERR command failed: {e}")

try:
    result = _line_type("read terr 1 2")
    print(f"Lowercase TERR command works: {result}")
except Exception as e:
    print(f"Lowercase TERR command failed: {e}")