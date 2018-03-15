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

    ``HarFileObj`` subclasses `dict`, and essentially stores a `list` of ``harpy.HeaderArrayObj`` as the value to the key ``"head_arrs"``. Each ``harpy.HeaderArrayObj`` corresponds to a header-array. If ``HarFileObj`` has been loaded from disk, each ``harpy.HeaderArrayObj`` corresponds to a header-array in that file.
    """

    def __init__(self, *args, filename: str=None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(filename, str):
            self["hfio"] = HarFileIO.readHarFileInfo(filename)
            self["hfio"].is_valid()

        self["head_arrs"] = []

    def getHeaderArrayNames(self):
        """
        :return: Returns the name of all ``harpy.HeaderArrayObj()`` stored in ``self``.
        """
        return [h["name"] for h in self["head_arrs"]]

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

    def getHeaderArrayObjs(self, ha_names: 'Union[None,str,List[str]]'=None):
        """
        Retrieve a `list` of ``harpy.HeaderArrayObj``.
        :param ha_names: The name or `list` of names of ``harpy.HeaderArrayObj``. If `None` is provided (the default) then all ``harpy.HeaderArrayObj`` are returned.
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

    # def readHeaderArrayObjs(self, filename: str, ha_names: 'Union[None,str,List[str]]' = None):
    def readHeaderArrayObjs(self, ha_names: 'Union[None,str,List[str]]'=None):
        """
        :param ha_names: Reads the header array objects with names ``ha_names``. If `None` (the default), read all header array objects. Reads from a file specified by a ``harpy.HarFileInfoObj`` automatically generated when creating ``self`` from the ``loadFromDisk()`` method, or by providing the ``filename`` argument at initialisation. The ``harpy.HarFileInfoObj`` is stored as the value matching the key ``"hfio"``.
        :return: `None`
        """

        if ha_names is None:
            ha_names = self["hfio"].getHeaderArrayNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        for ha_name in ha_names:
            self["head_arrs"].append(HarFileIO.readHeader(self["hfio"], ha_name))

        # HarFileIO.readHeaderArraysFromFile(filename=filename, ha_names=ha_names)

    def writeToDisk(self, filename: str=None, ha_names: 'Union[None,str,List[str]]'=None):
        """
        :param filename: If provided, writes to ``filename`` instead of overwriting the file read to.
        :return:
        """

        if ha_names is None:
            ha_names = self.getHeaderArrayNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        ha_to_write = self.getHeaderArrayObjs(ha_names)

        if filename is None:
            filename = self["hfio"]["file"]

        HarFileIO.writeHeaders(filename, ha_to_write)

    def removeHeaderArrayObjs(self, ha_names: 'Union[str,List[str]]'):

        if isinstance(ha_names, str):
            ha_names = [ha_names]

        for ha_name in ha_names:
            idx = self.getHeaderArrayObjIdx(ha_name)
            return self["head_arrs"].pop(idx)

    def addHeaderArrayObjs(self, ha_objs: 'Union[HeaderArrayObj,List[HeaderArrayObj]]'=None) -> None:
        """
        :param ha_objs:
        :return:
        """

        if isinstance(ha_objs, HeaderArrayObj):
            ha_objs = [ha_objs]

        for ha_obj in ha_objs:
            if ha_obj.is_valid():
                self.addHeaderArrayObj(ha_obj)

        return None

    def addHeaderArrayObj(self, ha_obj: HeaderArrayObj, idx: int=None):
        """
        :param ha_obj:
        :param idx:
        :return:
        """

        if idx is None:
            idx = len(self["head_arrs"])

        self["head_arrs"].insert(idx, ha_obj)


    @staticmethod
    def loadFromDisk(filename: str) -> 'HarFileObj':
        """
        Loads a HAR file into memory, returning a HarFileObj.
        :param filename:
        :return:
        """

        hfo = HarFileObj(filename=filename)
        hfo.readHeaderArrayObjs()
        # hfo.readHeaderArrayObjs(filename=filename)

        return hfo
