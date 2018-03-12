"""
Created on Mar 12 09:53:27 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

from typing import Union, List

from .har_file_io import HarFileIO

class HarFileObj(dict):
    """
    HAR file memory object - the equivalent of a HAR file stored in memory.
    """

    def __init__(self, *args, filename: str=None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(filename, str):
            self["hfio"] = HarFileIO.readHarFileInfo(filename)
            self["hfio"].is_valid()

        self["head_arrs"] = []

    def getHeaderNames(self):
        return [h["name"] for h in self["head_arrs"]]

    def getHeaderArrayObjIdx(self, ha_name):
        """
        :param ha_name: Name of Header Array.
        :return int: The `list` index of the header array.
        """
        for idx, hao in enumerate(self["head_arrs"]):
            if hao["name"] == ha_name:
                return idx
        else:
            # print(self["head_arrs"])
            raise ValueError("HeaderArray '%s' does not exist in HarFileObj." % (ha_name))

    def getHeaderArrayObjs(self, ha_names: Union[str, List[str]]):
        """**Not implemented**.
        Retrieve a HeaderArrayMemObj. If the object has not been read into memory, this will initiate a read operation."""

        if isinstance(ha_names, str):
            ha_names = [ha_names]

        ha_objs = []
        for ha_name in ha_names:
            idx = self.getHeaderArrayObjIdx(ha_name=ha_name)

            if self["head_arrs"][idx].is_valid():
                ha_objs.append(self["head_arrs"][idx])
            else:
                self.readHeaderArrayObjs(ha_names=ha_name)
                ha_objs.append(self.getHeaderArrayObjs(ha_name))
        return ha_objs

    def readHeaderArrayObjs(self, ha_names: Union[None, str, List[str]]=None):
        """Reads the header array objects with names ``ha_names``. If `None` (the default), read all header array objects.
        """

        if ha_names is None:
            ha_names = self["hfio"].getHeaderNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        for ha_name in ha_names:
            self["head_arrs"].append(HarFileIO.readHeader(self["hfio"], ha_name))

    def writeToDisk(self, filename: str=None, ha_names: Union[None, str, List[str]]=None):
        """
        :param filename: If provided, writes to ``filename`` instead of overwriting the file read to.
        :return:
        """

        if ha_names is None:
            ha_names = self.getHeaderNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        ha_to_write = self.getHeaderArrayObjs(ha_names)

        if filename is None:
            filename = self["hfio"]["file"]

        HarFileIO._writeHeaders(filename, ha_to_write)

        # for ha_name in ha_names:
        #     self["head_arrs"].append(HarFileIO.readHeader(self["hfio"], ha_name))


    # def write_HAR_File(self):
    #     """
    #     Write the content of the HAR object to the file associated with it
    #
    #     :return:
    #     """
    #     for name in self._HeaderList:
    #         if not name in self._HeaderDict:
    #             self.getHeader(name)
    #     self.f.seek(0)
    #     self.f.truncate()
    #     for name in self._HeaderList:
    #         Header=self._HeaderDict[name]
    #         if Header.is_valid:
    #             Header.HeaderToFile(self.f)
    #     self.f.f.flush()

    @staticmethod
    def loadFromDisk(filename: str) -> 'HarFileObj':
        """
        Loads a HAR file into memory, returning a HarFileObj.
        :param filename:
        :return:
        """

        hfmo = HarFileObj(filename=filename)
        # ha_names = hfmo["hfio"].getHeaderNames()
        hfmo.readHeaderArrayObjs()

        return hfmo
