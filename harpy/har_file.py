"""
.. autoclass: HarFileObj
    :members:

Created on Mar 12 09:53:27 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

from .har_file_io import HarFileIO
from .header_array import HeaderArrayObj

class HarFileObj(dict):
    """
    HAR file object - essentially a memory representation of a HAR file.

    ``HarFileObj`` subclasses `dict`, and stores a `list` of ``harpy.HeaderArrayObj`` in ``self``. Each ``harpy.HeaderArrayObj`` corresponds to a header-array. If ``HarFileObj`` is provided with ``filename``, then the header-arrays in that file will be loaded - i.e. each ``harpy.HeaderArrayObj``  in ``self`` will correspond to a header-array in that file.
    """

    def __init__(self, *args, filename: str=None, **kwargs):
        super().__init__(*args, **kwargs)

        self["head_arrs"] = []

        if isinstance(filename, str):
            self = HarFileObj.loadFromDisk(filename)


    def getHeaderArrayNames(self):
        """
        :return: Returns the name of all ``harpy.HeaderArrayObj()`` stored with ``self``.
        """
        return [h["name"] for h in self["head_arrs"]]

    def getRealHeaderArrayNames(self):
        """
        :return: Returns only the names of arrays of type 2D or 7D - i.e. multi-dimensional header arrays of floating point numbers.
        """

        return [h["name"] for h in self["head_arrs"] if h["data_type"] in ["RE"]]


    def getHeaderArrayObjIdx(self, ha_name):
        """
        :param ha_name: Name of Header Array.
        :return int: The `list` index of the ``harpy.HeaderArrayObj``.
        """
        for idx, hao in enumerate(self["head_arrs"]):
            if hao["name"] == ha_name:
                return idx
        else:
            raise ValueError("HeaderArrayObj '%s' does not exist in HarFileObj. A possible cause is that the HeaderArrayObj has not been read into memory." % (ha_name))

    def getHeaderArrayObj(self, ha_name: str):
        """
        Retrieve a single ``harpy.HeaderArrayObj``.

        :param ha_name: The ``"name"`` of the ``harpy.HeaderArrayObj``.
        :return: A ``harpy.HeaderArrayObj``.
        """

        if not isinstance(ha_name, str):
            raise TypeError("'ha_name' must be a string.")

        idx = self.getHeaderArrayObjIdx(ha_name=ha_name)

        return self["head_arrs"][idx]

    def getHeaderArrayObjs(self, ha_names=None):
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
            ha_objs.append(self.getHeaderArrayObj(ha_name))
        return ha_objs

    def readHeaderArrayObjs(self, filename: str, ha_names = None):
        """
         Reads the header array objects with names ``ha_names`` from ``filename``. If `None` (the default), read all header array objects. `harpy.HeaderArrayObj` are stored in ``self`` and can be retrieved with the ``self.getHeaderArrayObjs()`` method.

        :param str filename:
        :param 'Union[None,str,List[str]]' ha_names:
        :return: `None`
        """

        self["head_arrs"] = HarFileIO.readHeaderArraysFromFile(filename=filename, ha_names=ha_names)

    def writeToDisk(self, filename: str, ha_names=None):
        """
        :param str filename: Writes `harpy.HeaderArrayObj` with ``ha_names`` to ``filename``. If ``ha_names`` is None, write all the `harpy.HeaderArrayObj` stored in ``self``.
        :param 'Union[None,str,List[str]]' ha_names: The names of the header arrays to write to ``filename``.
        """

        if ha_names is None:
            ha_names = self.getHeaderArrayNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        ha_to_write = self.getHeaderArrayObjs(ha_names)

        HarFileIO.writeHeaders(filename, ha_to_write)

    def removeHeaderArrayObjs(self, ha_names):
        """
        :param 'Union[str,List[str]]' ha_names: Remove one or more `harpy.HeaderArrayObj` from ``self``.
        """

        if isinstance(ha_names, str):
            ha_names = [ha_names]

        for ha_name in ha_names:
            idx = self.getHeaderArrayObjIdx(ha_name)
            return self["head_arrs"].pop(idx)

    def addHeaderArrayObjs(self, ha_objs) -> None:
        """
        :param 'Union[HeaderArrayObj,List[HeaderArrayObj]]' ha_objs: Add one or more `harpy.HeaderArrayObj` to ``self``.
        """

        if isinstance(ha_objs, HeaderArrayObj):
            ha_objs = [ha_objs]

        for ha_obj in ha_objs:
            if ha_obj.is_valid():
                self.addHeaderArrayObj(ha_obj)

        return None

    def addHeaderArrayObj(self, ha_obj: HeaderArrayObj, idx: int=None):
        """
        :param ha_obj: A `harpy.HeaderArrayObj` object.
        :param idx: The index of ``self["head_arrs"]`` at which to insert ``ha_obj``.
        """

        if idx is None:
            idx = len(self["head_arrs"])

        self["head_arrs"].insert(idx, ha_obj)


    @staticmethod
    def loadFromDisk(filename: str) -> 'HarFileObj':
        """
        Loads a HAR file into memory, returning a HarFileObj.

        :param filename: The name of the file to load.
        :return:
        """

        hfo = HarFileObj()
        hfo.readHeaderArrayObjs(filename=filename)

        return hfo
