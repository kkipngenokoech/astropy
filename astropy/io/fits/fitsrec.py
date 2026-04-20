# Licensed under a 3-clause BSD style license - see PYFITS.rst

import sys
import warnings
from collections import OrderedDict

import numpy as np

from astropy.io.fits.column import (
    FITS2NUMPY,
    NUMPY2FITS,
    _AsciiColDefs,
    _FormatP,
    _FormatQ,
    _VLF,
    _get_index,
    _parse_tdisp_format,
    _scalar_to_format_string,
    _VarLenArrayType,
)
from astropy.io.fits.util import (
    _is_int,
    _is_pseudo_integer,
    _pseudo_zero,
    decode_ascii,
    encode_ascii,
)
from astropy.utils import lazyproperty


__all__ = ['FITS_rec', 'FITS_record']


# mapping from FITS format code to numpy format code
# this does not handle X format (bit) or P format (variable length arrays)
# TODO: Numpy does not support complex256 (i.e. 'c32') or float128 (i.e 'f16')
# TODO: For now, we use complex128 and float64 instead.
# TODO: Check for platform dependency
FITS_RECORD_FORMATS = {
    'L': 'bool',
    'B': 'uint8',
    'I': 'int16',
    'J': 'int32',
    'K': 'int64',
    'E': 'float32',
    'D': 'float64',
    'C': 'complex64',
    'M': 'complex128',
    'A': 'a',
    'X': 'uint8',
    'P': 'int32',
    'Q': 'int64'
}


class FITS_record:
    """
    FITS record class.

    `FITS_record` is used to access records of the `FITS_rec` object.
    This will allow us to deal with scaled columns.  It also handles
    conversion/scaling of columns.

    The `FITS_record` class expects a `FITS_rec` object as input.
    """

    def __init__(self, input, row=0, start=None, end=None, step=None,
                 base=None, **kwargs):
        """
        Parameters
        ----------
        input : array
           The array to wrap.

        row : int, optional
           The starting logical row of the array.

        start : int, optional
           The starting column in the row associated with this object.
           Used for subsetting the columns of the FITS_rec object.

        end : int, optional
           The ending column in the row associated with this object.
           Used for subsetting the columns of the FITS_rec object.
        """

        # For backward compatibility...
        if isinstance(input, np.ndarray) and input.dtype.names is not None:
            input = FITS_rec(input)

        self.array = input
        self.row = row
        if base:
            width = len(base)
        else:
            width = self.array._nfields

        # Set the slice attributes
        s = slice(start, end, step)
        indices = s.indices(width)
        self.start, self.end, self.step = indices

        # Set the base, if given
        self.base = base

    def __getitem__(self, key):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)

            if indx < self.start or indx >= self.end:
                raise KeyError(f"Key '{key}' does not exist.")

            # We have to adjust the index
            indx = (indx - self.start) // self.step

            return self.array.field(indx)[self.row]
        elif isinstance(key, slice):
            return type(self)(self.array, self.row, key.start, key.stop,
                              key.step, self.base)
        else:
            indx = self.start + key * self.step
            if indx >= self.end:
                raise IndexError('Index out of range')
            return self.array.field(indx)[self.row]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            indx = _get_index(self.array.names, key)

            if indx < self.start or indx >= self.end:
                raise KeyError(f"Key '{key}' does not exist.")

            indx = (indx - self.start) // self.step
            self.array.field(indx)[self.row] = value
        elif isinstance(key, slice):
            for indx in range(*key.indices(len(self))):
                self[indx] = value
        else:
            indx = self.start + key * self.step
            if indx >= self.end:
                raise IndexError('Index out of range')
            self.array.field(indx)[self.row] = value

    def __getslice__(self, start, end):
        return self[slice(start, end)]

    def __len__(self):
        return len(range(self.start, self.end, self.step))

    def __repr__(self):
        # Mimic the behavior of numpy record arrays
        return repr(tuple(self))

    def field(self, key):
        return self[key]


class FITS_rec(np.recarray):
    """
    FITS record array class.

    `FITS_rec` is the data object used by the `TableHDU` class.
    `FITS_rec` is a subclass of `numpy.recarray`, so we can use all
    of the available record array functionality.  It also includes
    the specialized functionality provided by `FITS_record`.

    See Also
    --------
    FITS_record

    Parameters
    ----------
    input : array
       The array to base the record on.

    """

    _record_type = FITS_record
    _character_as_bytes = False

    def __new__(subtype, input, character_as_bytes=False):
        """
        Construct a FITS record array from a numpy array.
        """

        if input is None:
            return None

        # make sure we are a subclass of the right type
        self = np.asarray(input).view(subtype)
        self._character_as_bytes = character_as_bytes
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return

        if isinstance(obj, FITS_rec):
            self._character_as_bytes = obj._character_as_bytes
        else:
            self._character_as_bytes = False

    def __getitem__(self, key):
        out = super().__getitem__(key)
        
        # For string columns, replace 'D' exponents with 'E' exponents
        if hasattr(out, 'dtype') and out.dtype.char in ('S', 'U'):
            if isinstance(out, np.ndarray) and out.ndim == 0:
                # Single string value
                str_val = str(out)
                if 'D+' in str_val or 'D-' in str_val:
                    str_val = str_val.replace('D+', 'E+').replace('D-', 'E-')
                    out = np.array(str_val, dtype=out.dtype)
            elif isinstance(out, np.ndarray):
                # Array of strings
                if out.dtype.char == 'S':
                    # Byte strings
                    out = out.astype('U')  # Convert to unicode for processing
                    out = np.char.replace(out, 'D+', 'E+')
                    out = np.char.replace(out, 'D-', 'E-')
                elif out.dtype.char == 'U':
                    # Unicode strings
                    out = np.char.replace(out, 'D+', 'E+')
                    out = np.char.replace(out, 'D-', 'E-')
        
        return out

    def __getslice__(self, start, end):
        return self[slice(start, end)]

    @classmethod
    def from_columns(cls, columns, nrows=0, fill=False, character_as_bytes=False):
        """
        Given either a `ColDefs` object, a sequence of `Column` objects, or
        another `FITS_rec` object, return a new `FITS_rec` object.
        """

        if isinstance(columns, cls):
            # Just return a copy
            return columns.copy()

        if not isinstance(columns, _AsciiColDefs):
            columns = ColDefs(columns)

        # Create the new array
        arraydef = columns._arrays
        if arraydef:
            # Use the arrays in the column definitions
            nrows = len(arraydef[0])
            arrays = []
            for arr in arraydef:
                if arr.dtype.char in ('S', 'U'):
                    # Handle D exponent replacement for string arrays
                    if arr.dtype.char == 'S':
                        arr = arr.astype('U')  # Convert to unicode
                    arr = np.char.replace(arr, 'D+', 'E+')
                    arr = np.char.replace(arr, 'D-', 'E-')
                arrays.append(arr)
            
            # Create structured array from the processed arrays
            dtype = [(name, arr.dtype, arr.shape[1:] if arr.ndim > 1 else ()) 
                     for name, arr in zip(columns.names, arrays)]
            
            data = np.empty(nrows, dtype=dtype)
            for name, arr in zip(columns.names, arrays):
                data[name] = arr
                
        else:
            # Create empty array with the right structure
            dtype = columns.dtype
            if nrows == 0:
                nrows = 1
            data = np.zeros(nrows, dtype=dtype)
            if fill:
                for idx, column in enumerate(columns):
                    data.field(idx)[:] = column.null

        return cls(data, character_as_bytes=character_as_bytes)

    @property
    def _nfields(self):
        return len(self.dtype.names) if self.dtype.names else 0

    @property
    def names(self):
        return self.dtype.names

    def field(self, key):
        """
        A view of a `Column`'s data as an array.
        """
        
        # Use the column index
        indx = _get_index(self.names, key)
        out = self._get_raw_data()[self.names[indx]]
        
        # Handle D exponent replacement for string fields
        if hasattr(out, 'dtype') and out.dtype.char in ('S', 'U'):
            if out.dtype.char == 'S':
                out = out.astype('U')  # Convert to unicode
            out = np.char.replace(out, 'D+', 'E+')
            out = np.char.replace(out, 'D-', 'E-')
            
        return out

    def _get_raw_data(self):
        """
        Returns the base array of this record array, without any special
        formatting applied.
        """
        # This just returns self as a plain recarray
        return self.view(np.recarray)


# Import ColDefs here to avoid circular imports
from astropy.io.fits.column import ColDefs