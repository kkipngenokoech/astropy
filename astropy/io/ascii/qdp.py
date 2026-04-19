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
    _command_re = r"[Rr][Ee][Aa][Dd] [TtSs][Ee][Rr][Rr](\s+[0-9]+)+"

    sep = delimiter
    if delimiter is None:
        sep = r"\s+"
    _new_re = rf"NO({sep}NO)+"
    _data_re = rf"({_decimal_re}|NO|[-+]?nan)({sep}({_decimal_re}|NO|[-+]?nan))*)"
    _type_re = rf"^\s*((?P<command>{_command_re})|(?P<new>{_new_re})|(?P<data>{_data_re})?\s*(\!(?P<comment>.*))?\s*$"
    _line_type_re = re.compile(_type_re)
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
    >>> lines = [line0, line1]
    >>> types, ncol = _get_type_from_list_of_lines(lines)
    >>> types[0]
    'comment'
    >>> types[1]
    'data,3'
    >>> ncol
    3
    >>> lines.append("23")
    >>> _get_type_from_list_of_lines(lines)
    Traceback (most recent call last):
        ...
    ValueError: Inconsistent number of columns
    """
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


def _get_lines_from_file(qdp_file):
    if "\n" in qdp_file:
        lines = qdp_file.split("\n")
    elif isinstance(qdp_file, str):
        with open(qdp_file) as fobj:
            lines = [line.strip() for line in fobj.readlines()]
    elif isinstance(qdp_file, Iterable):
        lines = qdp_file
    else:
        raise ValueError("invalid value of qdb_file")

    return lines


def _interpret_err_lines(err_specs, ncols, names=None):
    """Give list of column names from the READ SERR and TERR commands.

    Parameters
    ----------
    err_specs : dict
        ``{'serr': [n0, n1, ...], 'terr': [n2, n3, ...]}``
        Error specifications for symmetric and two-sided errors
    ncols : int
        Number of data columns

    Other Parameters
    ----------------
    names : list of str
        Name of data columns (defaults to ['col1', 'col2', ...]), _not_
        including error columns.

    Returns
    -------
    colnames : list
        List containing the column names. Error columns will have the name
        of the main column plus ``_err`` for symmetric errors, and ``_perr``
        and ``_nerr`` for positive and negative errors respectively

    Examples
    --------
    >>> col_in = ['MJD', 'Rate']
    >>> cols = _interpret_err_lines(None, 2, names=col_in)
    >>> cols[0]
    'MJD'
    >>> err_specs = {'terr': [1], 'serr': [2]}
    >>> ncols = 5
    >>> cols = _interpret_err_lines(err_specs, ncols, names=col_in)
    >>> cols[0]
    'MJD'
    >>> cols[2]
    'MJD_nerr'
    >>> cols[4]
    'Rate_err'
    >>> _interpret_err_lines(err_specs, 6, names=col_in)
    Traceback (most recent call last):
        ...
    ValueError: Inconsistent number of input colnames
    """
    colnames = ["" for i in range(ncols)]
    if err_specs is None:
        serr_cols = terr_cols = []

    else:
        # I don't want to empty the original one when using `pop` below
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


def _get_tables_from_qdp_file(qdp_file, input_colnames=None, delimiter=None):
    """Get all tables from a QDP file.

    Parameters
    ----------
    qdp_file : str
        Input QDP file name

    Other Parameters
    ----------------
    input_colnames : list of str
        Name of data columns (defaults to ['col1', 'col2', ...]), _not_
        including error columns.
    delimiter : str
        Delimiter for the values in the table.

    Returns
    -------
    list of `~astropy.table.Table`
        List containing all the tables present inside the QDP file
    """
    lines = _get_lines_from_file(qdp_file)
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

            if err_specs != {}:
                warnings.warn(
                    "This file contains multiple command blocks. Please verify",
                    AstropyUserWarning,
                )
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
                    values.append(np.ma.masked)
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
                new_table = Table(names=colnames, rows=current_rows)
                new_table.meta["initial_comments"] = initial_comments.strip().split(
                    "\n"
                )
                new_table.meta["comments"] = comment_text.strip().split("\n")
                # Reset comments
                comment_text = ""
                table_list.append(new_table)
                current_rows = None
            continue

    # At the very end, if there is still a table being written, let's save
    # it to the table_list
    if current_rows is not None:
        new_table = Table(names=colnames, rows=current_rows)
        new_table.meta["initial_comments"] = initial_comments.strip().split("\n")
        new_table.meta["comments"] = comment_text.strip().split("\n")
        table_list.append(new_table)

    return table_list


def _understand_err_col(colnames):
    """Get which column names are error columns.

    Examples
    --------
    >>> colnames = ['a', 'a_err', 'b', 'b_perr', 'b_nerr']
    >>> serr, terr = _understand_err_col(colnames)
    >>> np.allclose(serr, [1])
    True
    >>> np.allclose(terr, [2])
    True
    >>> serr, terr = _understand_err_col(['a', 'a_nerr'])
    Traceback (most recent call last):
    ...
    ValueError: Missing positive error...
    >>> serr, terr = _understand_err_col(['a', 'a_perr'])
    Traceback (most recent call last):
    ...
    ValueError: Missing negative error...
    """
    shift = 0
    serr = []
    terr = []

    for i, col in enumerate(colnames):
        if col.endswith("_err"):
            # The previous column, but they're numbered from 1!
            # Plus, take shift into account
            serr.append(i - shift)
            shift += 1
        elif col.endswith("_perr"):
            terr.append(i - shift)
            if len(colnames) == i + 1 or not colnames[i + 1].endswith("_nerr"):
                raise ValueError("Missing negative error")
            shift += 2
        elif col.endswith("_nerr") and not colnames[i - 1].endswith("_perr"):
            raise ValueError("Missing positive error")
    return serr, terr


def _read_table_qdp(qdp_file, names=None, table_id=None, delimiter=None):
    """Read a table from a QDP file.

    Parameters
    ----------
    qdp_file : str
        Input QDP file name

    Other Parameters
    ----------------
    names : list of str
        Name of data columns (defaults to ['col1', 'col2', ...]), _not_
        including error columns.

    table_id : int, optional
        Index of the table to read from the QDP file. QDP files can contain
        multiple tables. Default is to read the first table (table_id=0)

    delimiter : str, optional
        Column delimiter. Default is whitespace.

    Returns
    -------
    table : `~astropy.table.Table`
        Output table

    """
    tables = _get_tables_from_qdp_file(qdp_file, input_colnames=names, delimiter=delimiter)
    if len(tables) == 0:
        raise ValueError("No tables found")

    if table_id is None:
        if len(tables) > 1:
            warnings.warn(
                "table_id not specified. Reading the first available table",
                AstropyUserWarning,
            )
        table_id = 0

    # Check if table_id is valid
    if table_id >= len(tables):
        raise ValueError(f"table_id should be lower than {len(tables)}")

    return tables[table_id]


def _write_table_qdp(table, filename=None, err_specs=None):
    """Write a table to a QDP file.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input table to be written
    filename : str
        Output QDP file name

    Other Parameters
    ----------------
    err_specs : dict
        Dictionary of the format {'serr': [1], 'terr': [2, 3]}, specifying
        which columns have symmetric and two-sided errors (see QDP format
        specification)
    """
    import io

    fobj = io.StringIO()

    if "initial_comments" in table.meta and table.meta["initial_comments"] != [""]:
        for line in table.meta["initial_comments"]:
            line = line.strip()
            if not line.startswith("!"):
                line = "!" + line
            print(line, file=fobj)

    if err_specs is None:
        serr_cols, terr_cols = _understand_err_col(table.colnames)
        if serr_cols != []:
            err_specs = {"serr": serr_cols}
        if terr_cols != []:
            if err_specs is None:
                err_specs = {"terr": terr_cols}
            else:
                err_specs["terr"] = terr_cols

    if err_specs is not None:
        for err_type, cols in err_specs.items():
            col_string = " ".join([str(c) for c in cols])
            print(f"READ {err_type.upper()} {col_string}", file=fobj)

    if "comments" in table.meta and table.meta["comments"] != [""]:
        for line in table.meta["comments"]:
            line = line.strip()
            if not line.startswith("!"):
                line = "!" + line
            print(line, file=fobj)

    colnames_to_print = []
    for col in table.colnames:
        if not (col.endswith("_err") or col.endswith("_perr") or col.endswith("_nerr")):
            colnames_to_print.append(col)

    colnames_to_print_all = []
    for col in colnames_to_print:
        colnames_to_print_all.append(col)
        if col + "_err" in table.colnames:
            colnames_to_print_all.append(col + "_err")
        elif col + "_perr" in table.colnames:
            colnames_to_print_all.append(col + "_perr")
            colnames_to_print_all.append(col + "_nerr")

    print("!" + "\t".join(colnames_to_print_all), file=fobj)

    for row in table:
        line = []
        for col in colnames_to_print_all:
            if np.ma.is_masked(row[col]):
                line.append("NO")
            elif np.isnan(row[col]):
                line.append("nan")
            else:
                line.append(str(row[col]))
        print("\t".join(line), file=fobj)

    lines = fobj.getvalue()
    fobj.close()

    if filename is not None:
        with open(filename, "w") as fobj:
            fobj.write(lines)

    return lines.split("\n")


class QDPSplitter(core.DefaultSplitter):
    """
    Split on space for QDP tables
    """

    delimiter = " "


class QDPHeader(basic.CommentedHeaderHeader):
    """
    Header that uses the :class:`astropy.io.ascii.basic.QDPSplitter`
    """

    splitter_class = QDPSplitter
    comment = "!"
    write_comment = "!"


class QDPData(basic.BasicData):
    """
    QDP table data reader
    """

    splitter_class = QDPSplitter
    write_spacer_lines = ["NO NO NO NO NO"]

    def process_lines(self, lines):
        """
        These lines will already have comments stripped
        """
        re_comment = re.compile(r"^\s*!")
        re_new_path = re.compile(r"^\s*NO(\s+NO)+\s*$")
        re_command = re.compile(r"^\s*[Rr][Ee][Aa][Dd]")
        filtered_lines = []
        for line in lines:
            if re_comment.match(line):
                continue
            if re_new_path.match(line):
                continue
            if re_command.match(line):
                continue
            filtered_lines.append(line)
        return filtered_lines


class QDP(basic.Basic):
    """QDP table format (http://heasarc.gsfc.nasa.gov/docs/software/ftools/others/qdp/qdp.html).

    Example::

        ! Initial comment line 1
        ! Initial comment line 2
        READ TERR 1
        READ SERR 3
        ! Table 0 comment
        !a a(pos) a(neg) b c ce d
        53000.5   0.25  -0.5   1  1.5  3.5 2
        54000.5   1.25  -1.5   2  2.5  4.5 3
        NO NO NO NO NO
        ! Table 1 comment
        !a a(pos) a(neg) b c ce d
        54000.5   2.25  -2.5   NO  3.5  5.5 5
        55000.5   3.25  -3.5   4  4.5  6.5 nan

    The input table above contains some initial comments, the error commands,
    then two tables.
    This file format can contain multiple tables, so by default the first
    table will be read. The ``table_id`` parameter, which defaults to 0
    (first table), can be used to read any other table in the file.

    Note that the QDP format is flexible, and QDP files not containing
    commands will also be read. Comments are specified via "!" and ignored.
    """

    _format_name = "qdp"
    _io_registry_format_aliases = ["qdp"]
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
        # Read full table with all QDP commands
        return _read_table_qdp(
            table, names=self.names, table_id=self.table_id, delimiter=self.delimiter
        )

    def write(self, table, output):
        return _write_table_qdp(table, output, err_specs=self.err_specs)