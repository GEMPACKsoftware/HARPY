from __future__ import print_function, absolute_import
from .HAR_IO import HAR_IO
from .Header import Header
from collections import OrderedDict
from copy import deepcopy
from typing import List

__docformat__ = 'restructuredtext en'


class HAR(object):
    """
    This class is for loading HAR files into memory, manipulating the loaded HAR file virtually, and then writing \
    the manipulated version to disk.
    """

    def __init__(self, fname, mode):
        """
        Connects the file fname to the HAR object and parsers the Headers on this file.
        If mode w is chosen, file can be edited.
        All headers on the file are registered with the HAR object. Upon write all Headers are rewritten.

        :param fname: name of the HAR file
        :param mode: "w" or "r" for write or read
        """
        self._HeaderDict = OrderedDict()
        self.f = HAR_IO(fname, mode)
        self._HeaderList = self.f.getHeaderNames()

    # def getFileName(self):
    #     return self.f.getFileName()

    def getHeader(self, name, getDeepCopy=True):
        # type: (str, bool) -> Header
        """
        Returns the Header with the name name from the file object associated with the HAR object.
        The default behaviour returns a pointer to the HEader object on the HAR file, i.e. modifying this
        object changes the associated object in the HAR object(not the file).
        If an independent copy is required, set getDeepCopy=True
        :param name: str
        :param getDeepCopy: bool
        :returns: Header: Header
        :rtype: Header
        """

        name=name.strip().upper()

        if name in self.f.getHeaderNames():
            self._HeaderDict[name] = Header.HeaderFromFile(name, self.f)
            try:
                assert(isinstance(self._HeaderDict[name], Header))
            except AssertionError:
                raise TypeError("Invalid return type - return type attempting to get " +
                                "header '%s' is of type %s (id:%d), not '%s' (id:%d)." % (name,
                                                                                          self._HeaderDict[name].__class__,
                                                                                          id(self._HeaderDict[name].__class__),
                                                                                          Header,
                                                                                          id(Header)))

        # if name not in self._HeaderList:
        else:
            print("Header " + name + " was not found on file " + self.getFileName())
            return None

        if getDeepCopy:
            return self._HeaderDict[name].copy_header()
        else:
            return self._HeaderDict[name]

    def getHeaderNames(self):
        """
        :return: A `list` of header names.
        """
        return self.f.getHeaderNames()

    def getHeaderNames(self):
        """
        :return: A `list` of header names.
        """
        return self.f.getHeaderNames()

    def HeaderNames(self):
        """
        Can be used to obtain a list of all Headers on the file associated with the HAR object

        :return: list(str)
        """
        # :type: () -> list[str]
        raise DeprecationWarning("Method has been deprecated. Use 'getHeaderNames()' instead.")
        return self._HeaderList

    # def removeHeader(self, name):
    #     """
    #     Unregisters a header from the HAR object in memory.
    #
    #     To write the HAR object in memory to disk, :func:`write_HAR_File` has to be invoked.
    #
    #     :param str|Header name: The name of the header to be removed
    #     :return: `None`
    #     """
    #     if isinstance(name, Header):
    #         name = Header._HeaderName
    #     assert isinstance(name, str)

        if name not in self._HeaderDict:
            raise ValueError("Header '%s' does not exist in memory." % name)

        self._HeaderDict.pop(name)

    def removeHeader(self,Header):
        """
        Unregisters a header from the HAR object. Note, that this will not affect the content on file
        but only the Headers associated with the object. To update the file :func:`write_HAR_File` has to be invoked

        :param Header: Header to be unregistered from teh HAR object
        :type Header: Header
        :return:
        """
        if Header._HeaderName in self._HeaderDict:
            self._HeaderList.remove(Header._HeaderName)

    def addHeader(self,Header,overwrite=False):
        """
        Registers a header from the HAR object. Note, that this will not affect the content on file
        but only the Headers associated with the object. To update the file :func:`write_HAR_File` has to be invoked

        :param Header: Header to be registered with HAR object
        :type Header: Header
        :param overwrite: If a header with the same name is registered, this decides whether to use the new or the old header
        :type overwrite: bool
        :return:
        """

        if Header._HeaderName in self._HeaderDict and not overwrite:
            #print ("Header with name '" + Header._HeaderName + "' already on file")
            return
        else:
            if not Header._HeaderName in self._HeaderList:
                self._HeaderList.append(Header.HeaderName)
            self._HeaderDict[Header.HeaderName]=Header

    def write_HAR_File(self):
        """
        Write the content of the HAR object to the file associated with it

        :return:
        """
        for name in self._HeaderList:
            if not name in self._HeaderDict:
                self.getHeader(name)
        self.f.seek(0)
        self.f.truncate()
        for name in self._HeaderList:
            Header=self._HeaderDict[name]
            if Header.is_valid:
                Header.HeaderToFile(self.f)
        self.f.f.flush()

    # @classmethod
    # def cmbhar(cls,inFileList,outfile):
    #     """
    #
    #
    #     :param inFileList: list of filenames whose content has to be combined
    #     :type: inFileList: List of strings
    #     :param outfile: output file name
    #     :type outfile: str
    #     :return:
    #     """
    #     cls=HAR(outfile,'w')
    #     harList=[]
    #     for myFile in inFileList:
    #         harList.append(HAR(myFile,'r'))
    #     refHAR=harList[0]
    #     for name in refHAR.HeaderNames():
    #         AllList=[refHAR.getHeader(name)]
    #         if not 'float' in str(AllList[0].DataObj.dtype):
    #             cls.addHeader(AllList[0])
    #         else:
    #             for hars in harList[1:]:
    #                 AllList.append(hars.getHeader(name))
    #             CmbedHeader=Header.concatenate(AllList,elemList=['File'+str(i) for i in range(0,len(inFileList))],
    #                                            headerName=name)
    #             cls.addHeader(CmbedHeader)
    #     return cls


    # @classmethod
    # def diffhar(cls, inFileList, outfile):
    #     """
    #     computes the running difference between a set of har files.
    #     Differences are always taken between consecutive entries in the inFileList
    #
    #     :param inFileList: list of filenames whose content will be diffed
    #     :type: inFileList: List of strings
    #     :param outfile: output file name
    #     :type outfile: str
    #     :return:
    #     """
    #     cls = HAR(outfile, 'w')
    #     harList = []
    #     for myFile in inFileList:
    #         harList.append(HAR(myFile, 'r'))
    #     refHAR = harList[0]
    #     for name in refHAR.HeaderNames():
    #         AllList = [refHAR.getHeader(name)]
    #         if not 'float' in str(AllList[0].DataObj.dtype):
    #             cls.addHeader(AllList[0])
    #         else:
    #             for hars in harList[1:]:
    #                 AllList.append(hars.getHeader(name))
    #             elemList=['File' + str(i+1) +"-"+str(i) for i in range(0, len(inFileList)-1)]
    #             CmbedHeader = Header.runningDiff(AllList, elemList=elemList,headerName=name)
    #             cls.addHeader(CmbedHeader)
    #     return cls



