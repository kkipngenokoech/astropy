# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains functions for reading and writing QDP files that are
supported by the `PlotDevice`_ plotting package, and are often used by
high energy astrophysics packages.

.. _PlotDevice: http://hea-www.harvard.edu/RD/plotdevice/
"""

import copy
import re
from contextlib import suppress

import numpy as np

from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

from . import core


__all__ = ['QDPSplitter', 'QDPHeader', 'QDP']


def _line_type(line):
    """
    Determine the type of a QDP line.
    
    Parameters
    ----------
    line : str
        A line from a QDP file
        
    Returns
    -------
    str
        The type of line: 'comment', 'command', 'data', or 'new'
    """
    line = line.strip()
    
    if not line:
        return 'new'
    
    if line.startswith('!'):
        return 'comment'
    
    # Check for QDP commands (case-insensitive)
    first_word = line.split()[0].upper()
    if first_word in ('READ', 'PLOT', 'SKIP', 'NO', 'SERR', 'TERR', 'XERR', 'YERR'):
        return 'command'
    
    return 'data'


class QDPSplitter(core.DefaultSplitter):
    """
    Split on space for QDP tables
    """
    delimiter = ' '


class QDPHeader(core.BaseHeader):
    """
    Header reader for QDP Tables.
    """
    def __init__(self):
        super().__init__()
        self.comment = "!"

    def process_lines(self, lines):
        """
        Generator to yield non-comment lines
        """
        for line in lines:
            line = line.strip()
            if line and not line.startswith(self.comment):
                yield line

    def update_meta(self, lines, meta):
        """
        Extract table-level comments and keywords.
        """
        # Process comments and commands
        for line in lines:
            line = line.strip()
            if line.startswith(self.comment):
                # This is a comment
                meta.setdefault('comments', []).append(line[1:].strip())
            elif _line_type(line) == 'command':
                # These are QDP commands - store them
                meta.setdefault('keywords', {})[line.split()[0].upper()] = ' '.join(line.split()[1:])


class QDP(core.BaseReader):
    """
    Class to read tables from QDP files

    Example::

      >>> from astropy.table import Table
      >>> from astropy.io import ascii
      >>> table = Table.read('data.qdp', format='ascii.qdp')

    """
    _format_name = 'qdp'
    _io_registry_can_write = True
    _description = 'Quick and Dandy Plotter'

    header_class = QDPHeader
    data_class = core.BaseData
    splitter_class = QDPSplitter

    def __init__(self, table_id=None, names=None, err_specs=None):
        super().__init__()
        self.table_id = table_id
        self.names = names
        self.err_specs = err_specs or {}

    def _guess_error_specs(self, lines):
        """
        Guess error specifications from the data lines.
        """
        err_specs = {}
        
        # Look for lines that might contain data
        data_lines = []
        for line in lines:
            line = line.strip()
            if _line_type(line) == 'data':
                # Try to split and see if it's numeric data
                parts = line.split()
                if len(parts) > 0:
                    try:
                        # Try to convert first element to float
                        float(parts[0])
                        data_lines.append(parts)
                    except ValueError:
                        continue
        
        return err_specs