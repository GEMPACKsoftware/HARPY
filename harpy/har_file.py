"""
.. autoclass: HarFileObj
    :members:

Created on Mar 12 09:53:27 2018
"""

from .har_file_io import HarFileIO, HarFileInfoObj
from .header_array import HeaderArrayObj
from collections import OrderedDict
from typing import TypeVar, List, Union
from os import path
import warnings
TypeHarFileObj = TypeVar('TypeHarFileObj', bound='HarFileObj')

class HarFileObj(object):
    """
    HAR file object - essentially a memory representation of a HAR file.

    ``HarFileObj``  stores a `list` of ``harpy.HeaderArrayObj`` in ``self``.
     Each ``harpy.HeaderArrayObj`` corresponds to a header-array.
     If ``HarFileObj`` is provided with ``filename``, then the header-arrays in that file will be loaded - i.e. each ``harpy.HeaderArrayObj``  in ``self`` will correspond to a header-array in that file.

    Access to the ``HarFileObj``  is provided in a dict like style, __getitem__, __set_item__, __del_item__ and __contains__ are implemented. Each can take list arguments as well and returns result as list.
    Note that all methods are case insesitive with respect to names of Headers.

    Example: given file ex.har with Headers HD1 and HD2

    >>> from harpy import HarFileObj
    >>> thisHar=HarFileObj("ex.har")
    >>> headersOnFile= thisHar.getHeaderArrayNames() # ["HD1","HD2"]
    >>> hd1Head=thisHar["HD1"] # obtain the HeaderArrayObj for HD1
    >>> del thisHar["HD1"] # remove HD1 from HarFile
    >>> print ("HD1" in thisHar)
    False
    >>> thisHAR.writeToDisk() # overwrites ex.har, now only contains HD2 as HD1 was deleted


    The complete list of attributes is:

    :param OrderedDict head_arrs: Returned/provided as a `list` of ``HeaderArrayObj`` defining all ``HeaderArrayObj`` associated with a file.
    :param HarFileInfoObj _hfi  : Basic info of the HAr file content. This is used in conjuction with head_arrs to permit on the fly reading of ``HeaderArrayObj`` and thus readuce the memory footprint.

    And the methods of ``HarFileObj`` are:
    """

    def __init__(self, filename: str=None):
        self._head_arrs = OrderedDict()
        self.filename=filename
        if isinstance(filename, str):
            if path.isfile(filename):
                self._hfi = HarFileIO.readHarFileInfo(filename)
            else:
                self._hfi = HarFileInfoObj(file=filename)

    def __getitem__(self, item : 'Union[str, List[str]]' ):
        if isinstance(item,str):
            return self._getHeaderArrayObj(item)
        elif isinstance(item,list):
            if not all([isinstance(myitem,str) for myitem in item]):
                raise TypeError("All items in item must be of type 'str'")
            return self._getHeaderArrayObjs(item)
        else:
            raise TypeError("item must be string or list of strings")


    def __setitem__(self, key: 'Union[str, List[str]]', value: 'Union[HeaderArrayObj, List[HeaderArrayObj]]'):
        if isinstance(key, str) and isinstance(value,HeaderArrayObj):
            self._addHeaderArrayObj(key, value)
        elif isinstance(key, list) and isinstance(value,list):
            if not all([isinstance(mykey,str) for mykey in key]):
                raise TypeError("All items in key must be of type 'str'")
            if not all([isinstance(myval,HeaderArrayObj) for myval in value]):
                raise TypeError("All items in value must be of type 'HeaderArrayObj'")
            self._addHeaderArrayObjs(key, value)
        else:
            raise TypeError("Only combination str-HeaderArrayObj or list(str)-list(HeaderArrayObj) permitted in __getitem__'")

        return None

    def __delitem__(self, key):
        if isinstance(key,str):
            if key.strip().upper() in self._head_arrs:
                del self._head_arrs[key.strip().upper()]
        elif isinstance(key,list):
            for mykey in key:
                if mykey in self._head_arrs:
                    del self[mykey]
        else:
            raise TypeError("key must be string or list of strings")

        return None

    def __contains__(self, key):
        if isinstance(key,str):
            return key.strip().upper() in self._head_arrs
        return False




    def getHeaderArrayNames(self):
        """
        :return: Returns the name of all ``harpy.HeaderArrayObj()`` stored with ``self``.
        """

        if not self._hfi.is_valid(fatal=False):
            warnings.warn("Har file "+self._hfi.filename+" has changed since last access, rereading information")
            self._hfi=HarFileObj(self._hfi.file)._hfi
            self._head_arrs=OrderedDict()

        return self._hfi.getHeaderArrayNames()

    def getRealHeaderArrayNames(self):
        """
        :return: Returns only the names of arrays of type 2D or 7D - i.e. multi-dimensional header arrays of floating point numbers.
        """

        if not self._hfi.is_valid():
            warnings.warn("Har file "+self._hfi.filename+" has changed since last access, rereading information")
            self._hfi = HarFileObj(self._hfi.file)._hfi
            self._head_arrs = OrderedDict()
        return [key for key,val in self._hfi.items() if val.data_type in ["RE","RL","2R"]]


    def _getHeaderArrayObj(self, ha_name: str):
        """
        Retrieve a single ``harpy.HeaderArrayObj``.

        :param ha_name: The ``"name"`` of the ``harpy.HeaderArrayObj``.
        :return: A ``harpy.HeaderArrayObj``.
        """

        if not self._hfi.is_valid(fatal=False):
            warnings.warn("Har file "+self._hfi.filename+" has changed since last access, rereading information")
            self._hfi = HarFileObj(self._hfi.file)._hfi
            self._head_arrs = OrderedDict()

        if not isinstance(ha_name, str):
            raise TypeError("'ha_name' must be a string.")

        upname=ha_name.strip().upper()
        if not upname in self._hfi:
            raise KeyError("HeaderArrayObj '%s' does not exist in HarFileObj." % ha_name)
        if not upname in self._head_arrs:
            hnames, haos=  HarFileIO.readHeaderArraysFromFile(self._hfi, ha_names=upname)
            self._head_arrs[upname]=haos[0]

        return self._head_arrs[upname]

    def _getHeaderArrayObjs(self, ha_names=None):
        """
        Retrieve a `list` of `harpy.HeaderArrayObj`.

        :param 'Union[None,str,List[str]]' ha_names: The name or `list` of names of ``harpy.HeaderArrayObj``. If `None` is provided (the default) then all ``harpy.HeaderArrayObj`` are returned.
        :return: `list` of ``harpy.HeaderArrayObj``.
        """

        if isinstance(ha_names, str):
            ha_names = [ha_names]
        elif ha_names is None:
            ha_names = self.getHeaderArrayNames()

        ha_objs = []
        for ha_name in ha_names:
            ha_objs.append(self._getHeaderArrayObj(ha_name))
        return ha_objs

    def _readHeaderArrayObjs(self, ha_names = None):
        """
         Reads the header array objects with names ``ha_names`` from ``filename``. If `None` (the default), read all header array objects. `harpy.HeaderArrayObj` are stored in ``self`` and can be retrieved with the ``self.getHeaderArrayObjs()`` method.

        :param 'Union[None,str,List[str]]' ha_names:
        """
        hnames, haos = HarFileIO.readHeaderArraysFromFile(self._hfi, ha_names=ha_names)
        self._head_arrs=OrderedDict(zip(hnames, haos))


    def writeToDisk(self, filename: str=None, ha_names=None):
        """
        :param str filename: Writes `harpy.HeaderArrayObj` with ``ha_names`` to ``filename``. If ``ha_names`` is None, write all the `harpy.HeaderArrayObj` stored in ``self``.
        :param 'Union[None,str,List[str]]' ha_names: The names of the header arrays to write to ``filename``.
        """
        if filename is None and self.filename is None:
            raise ValueError("No filename specified in write or upon creation, use writeToDisk(filename=YOURFILENAME)")
        if filename is None:
            filename=self.filename
        if ha_names is None:
            ha_names = self.getHeaderArrayNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        ha_to_write = self._getHeaderArrayObjs(ha_names)

        HarFileIO.writeHeaders(filename, ha_names, ha_to_write)
        self._hfi.updateMtime()

    def _removeHeaderArrayObjs(self, ha_names):
        """
        TODO: its more of a pop, not a remove, maybe rename
        :param 'Union[str,List[str]]' ha_names: Remove one or more `harpy.HeaderArrayObj` from ``self``.
        """

        if isinstance(ha_names, str):
            ha_names = [ha_names]

        outlist=self._getHeaderArrayObjs(ha_names)

        for ha_name in ha_names:
            if ha_name.strip().upper() in self._hfi:
                del self._hfi._ha_infos[ha_name.strip().upper()]
            if ha_name.strip().upper() in self._head_arrs:
                del self._head_arrs[ha_name.strip().upper()]
        return outlist

    def _addHeaderArrayObjs(self, hnames, ha_objs) -> None:
        """
        :param 'Union[HeaderArrayObj,List[HeaderArrayObj]]' ha_objs: Add one or more `harpy.HeaderArrayObj` to ``self``.
        """

        if isinstance(ha_objs, HeaderArrayObj):
            ha_objs = [ha_objs]
        if isinstance(hnames, str):
            hnames = [hnames]

        for hname, ha_obj in zip(hnames,ha_objs):
            if ha_obj.is_valid():
                self._addHeaderArrayObj(hname, ha_obj)

        return None

    def _addHeaderArrayObj(self, hname : str, ha_obj: HeaderArrayObj):
        """
        :param ha_obj: A `harpy.HeaderArrayObj` object.
        """

        if len(hname.strip()) > 4:
            raise HarFileObj.InvalidHeaderArrayName("Name of Header too long")

        self._hfi.addHAInfo(hname.strip().upper(),0,0)
        self._head_arrs[hname.strip().upper()]= ha_obj


    @staticmethod
    def _loadFromDisk(filename: str, ha_names: list = None) -> TypeHarFileObj:
        """Loads a HAR file into memory, returning a HarFileObj.

        :param filename: The name of the file to load.
        :param ha_names: If provided, only reads headers with the names matching the strings contained in this list. By default, all header arrays are read.
        :return "HarFileObj": Returns ``HarFileObj`` with
        """

        hfo = HarFileObj(filename=filename)
        hfo._readHeaderArrayObjs(ha_names=ha_names)

        return hfo


    class InvalidHeaderArrayName(ValueError):
        """Raised if header array name is not exactly four (alphanumeric) characters long."""
        pass
