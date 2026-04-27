# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains functions for reading and writing QDP tables that are
not meant to be used directly, but instead are available as readers/writers in
`astropy.table`. See :ref:`astropy:table_io` for more details.
"""
import copy
import re
import warnings
from collections.abc import Iterable

import numpy as np

from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from . import basic, core


def _line_type(line, delimiter=None):
    """Interpret a QDP file line.

    Parameters
    ----------
    line : str
        a single line of the file

    Returns
    -------
    type : str
        Line type: "comment", "command", or "data"

    Examples
    --------
    >>> _line_type("READ SERR 3")
    'command'
    >>> _line_type(" \\n    !some gibberish")
    'comment'
    >>> _line_type("   ")
    'comment'
    >>> _line_type(" 21345.45")
    'data,1'
    >>> _line_type(" 21345.45 1.53e-3 1e-3 .04 NO nan")
    'data,6'
    >>> _line_type(" 21345.45,1.53e-3,1e-3,.04,NO,nan", delimiter=',')
    'data,6'
    >>> _line_type(" 21345.45 ! a comment to disturb")
    'data,1'
    >>> _line_type("NO NO NO NO NO")
    'new'
    >>> _line_type("NO,NO,NO,NO,NO", delimiter=',')
    'new'
    >>> _line_type("N O N NOON OON O")
    Traceback (most recent call last):
        ...
    ValueError: Unrecognized QDP line...
    >>> _line_type(" some non-comment gibberish")
    Traceback (most recent call last):
        ...
    ValueError: Unrecognized QDP line...
    """
    _decimal_re = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
    _command_re = r"READ [TS]ERR(\s+[0-9]+)+"

    sep = delimiter
    if delimiter is None:
        sep = r"\s+"
    _new_re = rf"NO({sep}NO)+"
    _data_re = rf"({_decimal_re}|NO|[-+]?nan)({sep}({_decimal_re}|NO|[-+]?nan))*)"
    _type_re = rf"^\s*((?P<command>{_command_re})|(?P<new>{_new_re})|(?P<data>{_data_re})?\s*(\!(?P<comment>.*))?)\s*$"
    _line_type_re = re.compile(_type_re, re.IGNORECASE)
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
    """Read through the list of QDP file lines and label each line by type.

    Parameters
    ----------
    lines : list
        List containing one file line in each entry

    Returns
    -------
    contents : list
        List containing the type for each line (see `line_type_and_data`)
    ncol : int
        The number of columns in the data lines. Must be the same throughout
        the file

    Examples
    --------
    >>> line0 = "! A comment"
    >>> line1 = "543 12 456.0"
    >>> line2 = "NO NO NO"
    >>> _get_type_from_list_of_lines([line0, line1, line2])
    (['comment', 'data,3', 'new'], 3)
    >>> lines = ["READ SERR 1 2", "1 0.5 1 0.5"]
    >>> _get_type_from_list_of_lines(lines)
    (['command', 'data,4'], 4)
    """
    ncol = None
    type_list = []
    for line in lines:
        line_type = _line_type(line, delimiter=delimiter)
        type_list.append(line_type)
        if line_type.startswith("data"):
            this_ncol = int(line_type.split(",")[1])
            if ncol is None:
                ncol = this_ncol
            elif ncol != this_ncol:
                raise ValueError(
                    f"Inconsistent number of columns: {ncol} vs {this_ncol}"
                )
    return type_list, ncol


def _get_lines_from_file(qdp_file):
    if hasattr(qdp_file, "read"):
        lines = qdp_file.read().splitlines()
    else:
        with open(qdp_file) as fh:
            lines = fh.read().splitlines()
    return lines


def _interpret_err_lines(err_specs, ncol, names=None):
    """Give list of column names from the READ SERR and TERR commands.

    Parameters
    ----------
    err_specs : dict
        ``{'serr': [n0, n1, ...], 'terr': [n2, n3, ...]}``
        Error specifications for symmetric and two-sided errors
    ncol : int
        Number of columns in the data
    names : list of str
        Column names. If not specified, defaults to ['col1', 'col2', ...]

    Returns
    -------
    colnames : list
        List containing the column names. Error columns will have the name
        of the main column plus ``_err`` for symmetric errors, and ``_perr``
        and ``_nerr`` for positive and negative errors respectively

    Examples
    --------
    >>> col_in = ['MJD', 'Rate']
    >>> ncol_in = 4
    >>> err_specs = {'terr': [1], 'serr': [2]}
    >>> _interpret_err_lines(err_specs, ncol_in, names=col_in)
    ['MJD', 'MJD_nerr', 'MJD_perr', 'Rate', 'Rate_err']
    """
    colnames = [f"col{i}" for i in range(1, ncol + 1)]
    if names is not None:
        colnames[: len(names)] = names

    for error_type, columns in err_specs.items():
        for c in columns:
            if c - 1 > len(colnames):
                raise ValueError(f"Error specification for non-existing column {c}")

    shift = 0
    for error_type, columns in err_specs.items():
        for c in sorted(columns):
            col_num = c - 1 + shift
            col_name = colnames[col_num]
            if error_type == "terr":
                colnames.insert(col_num + 1, col_name + "_nerr")
                colnames.insert(col_num + 2, col_name + "_perr")
                shift += 2
            elif error_type == "serr":
                colnames.insert(col_num + 1, col_name + "_err")
                shift += 1

    return colnames


def _get_tables_from_qdp_file(qdp_file, names=None, table_id=None, delimiter=None):
    """Get all tables from a QDP file.

    Parameters
    ----------
    qdp_file : str
        Input QDP file name

    Other Parameters
    ----------------
    names : list of str, optional
        Name of data columns (defaults to ['col1', 'col2', ...])
    table_id : int, optional
        Number of the table to read from the file (default: read all tables)
    delimiter : str, optional
        Column delimiter

    Returns
    -------
    tables : list of `~astropy.table.Table`
        List containing all the tables present in the file
    """
    lines = _get_lines_from_file(qdp_file)
    contents, ncol = _get_type_from_list_of_lines(lines, delimiter=delimiter)

    table_list = []
    err_specs = {"terr": [], "serr": []}

    comment_text = ""
    initial_comments = True
    command_lines = ""
    current_table_lines = []
    current_table_comments = []
    for line, line_type in zip(lines, contents):
        line = line.strip()
        if line_type == "comment":
            comment_text += line + "\n"
            if not initial_comments:
                current_table_comments.append(line)
        elif line_type == "command":
            # This is a command. If we already have some data,
            # we need to flush the previous table.
            if len(current_table_lines) > 0:
                colnames = _interpret_err_lines(err_specs, ncol, names=names)
                table_list.append(
                    _lines_to_table(
                        current_table_lines,
                        ncol=ncol,
                        colnames=colnames,
                        delimiter=delimiter,
                    )
                )
                current_table_lines = []
                current_table_comments = []

            if line.upper().startswith("READ SERR"):
                err_specs["serr"] = [int(c) for c in line.split()[2:]]
            elif line.upper().startswith("READ TERR"):
                err_specs["terr"] = [int(c) for c in line.split()[2:]]
            command_lines += line + "\n"
            initial_comments = False
        elif line_type.startswith("data"):
            initial_comments = False
            current_table_lines.append(line)
        elif line_type == "new":
            # This is a new table
            if len(current_table_lines) > 0:
                colnames = _interpret_err_lines(err_specs, ncol, names=names)
                table_list.append(
                    _lines_to_table(
                        current_table_lines,
                        ncol=ncol,
                        colnames=colnames,
                        delimiter=delimiter,
                    )
                )
                current_table_lines = []
                current_table_comments = []
            initial_comments = False

    # If there are remaining lines, create a table for them
    if len(current_table_lines) > 0:
        colnames = _interpret_err_lines(err_specs, ncol, names=names)
        table_list.append(
            _lines_to_table(
                current_table_lines,
                ncol=ncol,
                colnames=colnames,
                delimiter=delimiter,
            )
        )

    # Add the comments and command lines as metadata
    for table in table_list:
        table.meta["comments"] = comment_text.split("\n")
        table.meta["keywords"] = {}
        if command_lines != "":
            table.meta["keywords"]["comments"] = command_lines.split("\n")

    if table_id is not None:
        if table_id >= len(table_list):
            raise ValueError(f"Table {table_id} not found")
        return [table_list[table_id]]

    return table_list


def _lines_to_table(lines, ncol=None, colnames=None, delimiter=None):
    """Read lines with data and return an astropy Table.

    Parameters
    ----------
    lines : list
        List containing one file line in each entry
    ncol : int, optional
        Number of columns in the data. If not specified, it will be
        determined from the first line of data
    colnames : list, optional
        List of column names
    delimiter : str, optional
        Column delimiter

    Returns
    -------
    table : `~astropy.table.Table`
        Output table
    """
    if ncol is None:
        ncol = len(lines[0].split(delimiter))

    if colnames is None:
        colnames = [f"col{i}" for i in range(1, ncol + 1)]

    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove comments
        if "!" in line:
            line = line[: line.index("!")]
        values = line.split(delimiter)
        if len(values) != ncol:
            raise ValueError(f"Inconsistent number of columns: {len(values)} vs {ncol}")
        data.append(values)

    # Convert to numpy array and handle NO/nan values
    data_array = np.array(data, dtype=object)
    
    # Create table with proper column types
    table_data = {}
    for i, colname in enumerate(colnames):
        col_data = data_array[:, i]
        # Convert NO and nan to np.nan, try to convert to float
        converted_col = []
        for val in col_data:
            if val.upper() == "NO" or val.lower() == "nan":
                converted_col.append(np.nan)
            else:
                try:
                    converted_col.append(float(val))
                except ValueError:
                    converted_col.append(val)
        table_data[colname] = converted_col

    return Table(table_data)


class QDPSplitter(core.DefaultSplitter):
    """Split on space for QDP tables."""

    delimiter = " "


class QDPHeader(basic.CommentedHeaderHeader):
    """Header that uses the :class:`astropy.io.ascii.basic.QDPSplitter`."""

    splitter_class = QDPSplitter
    comment = "!"
    write_comment = "!"


class QDPData(basic.BasicData):
    """QDP table data reader."""

    splitter_class = QDPSplitter
    fill_values = [(core.masked, "NO")]
    comment = "!"
    write_comment = "!"


class QDP(basic.Basic):
    """Quick and Dandy Plotter format table.

    Example::

        ! Initial comment line 1
        ! Initial comment line 2
        READ SERR 1 2
        READ TERR 3
        ! Table 0 comment
        1 0.5 1 0.5 2 0.1 2.4 0.3
        2 1.5 2 1.5 3 0.2 3.4 0.7
        ! Table 1 comment
        NO NO NO NO NO NO NO NO
        11 1.1 11 1.1 13 1.3 13.4 1.4
        12 1.2 12 1.2 14 1.4 14.4 1.5

    The input table above contains some initial comments, the error commands,
    then two tables.
    This file format can contain multiple tables, separated by a line full
    of ``NO`` values.
    The comments at the beginning of each table are appended to the table
    ``meta``.

    The QDP format specification requires that the command sections come
    before the data, and that error commands are specified before
    ``READ SERR`` and ``READ TERR`` commands, which indicate which columns
    contain symmetric and two-sided errors, respectively.
    These commands accept a list of column numbers (starting from 1).
    """

    _format_name = "qdp"
    _io_registry_can_write = True
    _io_registry_suffix = ".qdp"
    _description = "Quick and Dandy Plotter"

    header_class = QDPHeader
    data_class = QDPData

    def __init__(self, table_id=None, names=None, delimiter=None):
        super().__init__()
        self.table_id = table_id
        self.names = names
        self.delimiter = delimiter

    def read(self, table):
        self.lines = self.inputter.get_lines(table, newline="\n")
        tables = _get_tables_from_qdp_file(
            self.lines, names=self.names, table_id=self.table_id, delimiter=self.delimiter
        )
        return tables[0]

    def write(self, table, output):
        """Write table as list of strings.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Input table data
        output : str
            Output file name
        """
        lines = []
        
        # Write comments
        if "comments" in table.meta:
            for comment in table.meta["comments"]:
                if comment.strip():
                    lines.append(f"! {comment}")
        
        # Write commands if present
        if "keywords" in table.meta and "comments" in table.meta["keywords"]:
            for command in table.meta["keywords"]["comments"]:
                if command.strip():
                    lines.append(command)
        
        # Write data
        for row in table:
            row_data = []
            for col in table.columns.values():
                val = row[col.name]
                if np.ma.is_masked(val) or np.isnan(val):
                    row_data.append("NO")
                else:
                    row_data.append(str(val))
            lines.append(" ".join(row_data))
        
        self.outputter.write(lines, output)


class QDPInputter(core.BaseInputter):
    def process_lines(self, lines):
        return lines


class QDPOutputter(core.TableOutputter):
    pass