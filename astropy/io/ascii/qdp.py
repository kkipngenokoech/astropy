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
    >>> line0 = "! A comment"
    >>> line1 = "543 12 456.0"
    >>> line2 = "543 13 456.0"
    >>> _get_type_from_list_of_lines([line0, line1, line2])
    (['comment', 'data,3', 'data,3'], 3)
    >>> line0 = "! A comment"
    >>> line1 = "543 12 456.0"
    >>> line2 = "543 13 456.0 456.0"
    >>> _get_type_from_list_of_lines([line0, line1, line2])
    Traceback (most recent call last):
        ...
    ValueError: Inconsistent number of columns
    """
    contents = []
    ncol = None
    for line in lines:
        line_type = _line_type(line, delimiter=delimiter)
        if line_type.startswith("data"):
            thisncol = int(line_type.split(",")[1])
            if ncol is None:
                ncol = thisncol
            elif ncol != thisncol:
                raise ValueError("Inconsistent number of columns")

        contents.append(line_type)
    return contents, ncol


def _get_lines_from_file(qdp_file):
    if hasattr(qdp_file, "read"):
        lines = qdp_file.read().splitlines()
    else:
        with open(qdp_file) as fh:
            lines = fh.read().splitlines()
    return lines


def _interpret_err_lines(err_specs, ncols, names=None):
    """Give list of column names from the READ SERR and TERR commands.

    Parameters
    ----------
    err_specs : dict
        ``{'serr': [n0, n1, ...], 'terr': [n2, n3, ...]}``
        Error specifications for symmetric and two-sided errors
    ncols : int
        Number of columns in the data
    names : list of str
        Column names. If not specified, defaults to ['col1', 'col2', ...]

    Returns
    -------
    colnames : list
        List containing the column names. Error columns will have the name
        of the main column plus ``_err`` for symmetric errors, and
        ``_perr`` (positive error) and ``_nerr`` (negative error) for
        two-sided errors.

    Examples
    --------
    >>> col_in = ['MJD', 'Rate']
    >>> cols = _interpret_err_lines({'terr': [1], 'serr': [2]}, 4, names=col_in)
    >>> cols
    ['MJD', 'MJD_nerr', 'MJD_perr', 'Rate', 'Rate_err']
    >>> _interpret_err_lines({'terr': [1], 'serr': [2]}, 4)
    ['col1', 'col1_nerr', 'col1_perr', 'col2', 'col2_err']
    """
    if names is not None:
        colnames = copy.deepcopy(names)
    else:
        colnames = [f"col{i}" for i in range(1, ncols + 1)]

    if "serr" in err_specs:
        for val in err_specs["serr"]:
            if val - 1 >= len(colnames):
                raise ValueError(f"Error specification refers to non-existing column {val}")
            colnames.insert(val, colnames[val - 1] + "_err")

    if "terr" in err_specs:
        for val in err_specs["terr"]:
            if val - 1 >= len(colnames):
                raise ValueError(f"Error specification refers to non-existing column {val}")
            colnames.insert(val, colnames[val - 1] + "_nerr")
            colnames.insert(val + 1, colnames[val - 1].replace("_nerr", "") + "_perr")

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
        Number of the table to be read from the file. If None, all tables
        are read.
    delimiter : str, optional
        Column delimiter. If not specified, whitespace is used.

    Returns
    -------
    tables : list of `~astropy.table.Table`
        List containing all the tables present in the file
    """
    lines = _get_lines_from_file(qdp_file)
    contents, ncol = _get_type_from_list_of_lines(lines, delimiter=delimiter)

    table_list = []
    err_specs = {}

    # Read the comments and commands
    command_lines = ""
    current_table = None
    for i, [line, content] in enumerate(zip(lines, contents)):
        line = line.strip()
        if content == "comment":
            continue
        elif content == "command":
            # This is a command
            command_lines += line.lower() + "\n"
            continue
        elif content.startswith("data") and current_table is None:
            # We are reading a table for the first time. Check if there are
            # error specs
            if re.search(r"read\s+serr", command_lines):
                serr_col = re.findall(r"read\s+serr\s+(\d+(?:\s+\d+)*)", command_lines)
                if serr_col:
                    err_specs["serr"] = [int(val) for val in serr_col[0].split()]
            if re.search(r"read\s+terr", command_lines):
                terr_col = re.findall(r"read\s+terr\s+(\d+(?:\s+\d+)*)", command_lines)
                if terr_col:
                    err_specs["terr"] = [int(val) for val in terr_col[0].split()]

            colnames = _interpret_err_lines(err_specs, ncol, names=names)
            current_table = Table(names=colnames)
            command_lines = ""

        if content.startswith("data"):
            values = []
            # Comments are also OK between data
            comment_text = ""
            if "!" in line:
                comment_text = line.split("!")[-1]
                line = line.split("!")[0]
            line = line.strip().split(sep=delimiter)
            for val in line:
                if val.upper() == "NO":
                    values.append(np.ma.masked)
                else:
                    # Understand if number is int or float
                    try:
                        values.append(int(val))
                    except ValueError:
                        values.append(float(val))
            current_table.add_row(values)
            if comment_text != "":
                current_table.meta["comments"] = current_table.meta.get("comments", []) + [
                    comment_text.strip()
                ]
        elif content == "new":
            # Save table and reset
            if current_table is not None:
                table_list.append(current_table)
            current_table = None
            err_specs = {}

    # At the end, if there is still a table being written, let's save it
    if current_table is not None:
        table_list.append(current_table)

    if table_id is not None:
        return table_list[table_id]
    else:
        return table_list


class QDPSplitter(core.DefaultSplitter):
    delimiter = None


class QDPHeader(basic.CommentedHeaderHeader):
    def __init__(self):
        super().__init__()
        self.comment = "!"

    def get_cols(self, lines):
        """Initialize the header Column objects from the table ``lines``.

        Based on the previously set Header attributes find or create the column names.
        Sets ``self.cols`` with the list of Columns.

        Parameters
        ----------
        lines : list
            List of table lines
        """
        # This function needs to do absolutely nothing, and it will crash
        # if it calls the parent class
        pass


class QDPData(basic.BasicData):
    splitter_class = QDPSplitter

    def __init__(self):
        super().__init__()
        self.comment = "!"


class QDP(basic.Basic):
    """Quick and Dandy Plotter.

    This format is often used by FTOOLS. It stores tables in a text format
    that supports errors on measurements and upper/lower limits
    (``astropy.table`` does not support upper and lower limits, so
    these are ignored by default).

    When writing, you can specify whether or not you want the errors to be
    explicitly written to the table.  You do this by setting
    ``err_specs`` to a dictionary mapping data column names to error column
    names.  The error can be symmetric or asymmetric.  Example::

        >>> from astropy.table import Table
        >>> import numpy as np
        >>> t = Table()
        >>> t['X'] = [1,2,3]
        >>> t['Y'] = [2.3, 4.5, 8.1]
        >>> t['Yerr'] = [0.2, 0.5, 0.1]
        >>> t['Yperr'] = [0.2, 0.5, 0.1]
        >>> t['Ynerr'] = [0.1, 0.3, 0.2]
        >>> t.write(filename, format='ascii.qdp',
        ...         err_specs={'Y': {'serr': 'Yerr'},
        ...                    'X': {'terr': ['Ynerr', 'Yperr']}})

    Note: When the input table has units, they are written to the QDP file
    and will be recovered when reading.

    """

    _format_name = "qdp"
    _io_registry_can_write = True
    _io_registry_suffix = ".qdp"
    _description = "Quick and Dandy Plotter"

    header_class = QDPHeader
    data_class = QDPData

    def __init__(self, table_id=None, names=None, err_specs=None, sep=None):
        super().__init__()
        self.table_id = table_id
        self.names = names
        self.err_specs = err_specs
        self.delimiter = sep

    def read(self, table):
        # Read and parse the table
        self.lines = self.inputter.get_lines(table, newline="\n")
        try:
            tables = _get_tables_from_qdp_file(
                self.lines, names=self.names, table_id=self.table_id, delimiter=self.delimiter
            )
        except ValueError as e:
            raise core.InconsistentTableError(str(e))
        return tables

    def write(self, lines):
        if self.delimiter is None:
            delimiter = " "
        else:
            delimiter = self.delimiter
        if self.err_specs is None:
            self.err_specs = {}

        lines_out = []
        if len(self.err_specs) > 0:
            # Collect error columns
            serr_cols = []
            terr_cols = []
            for colname, err_specs in self.err_specs.items():
                if "serr" in err_specs:
                    serr_cols.append(self.cols.keys().index(colname) + 1)
                if "terr" in err_specs:
                    terr_cols.append(self.cols.keys().index(colname) + 1)

            if len(serr_cols) > 0:
                lines_out.append("READ SERR " + " ".join([str(val) for val in serr_cols]))
            if len(terr_cols) > 0:
                lines_out.append("READ TERR " + " ".join([str(val) for val in terr_cols]))

        for line in lines:
            lines_out.append(delimiter.join(line))
        lines_out.append("")
        return lines_out