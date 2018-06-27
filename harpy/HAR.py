from __future__ import print_function, absolute_import
from .HAR_IO import HAR_IO
from .Header import Header
import os
from copy import deepcopy

__docformat__ = 'restructuredtext en'


class HAR(object):
    """
    HAR class contains all functions to operate on a HAR files.

    This class is used for Assembling, Reading, Writing, combining, diffing,... of complete Header files.
    All data is stored in Header objects

    """

    def __init__(self, fname, mode):
        """
        Connects the file fname to the HAR object and parsers the Headers on this file.
        If mode w is chosen, file can be edited.
        All headers on the file are registered with the HAR object. Upon write all Headers are rewritten.

        :param fname: name of the HAR file
        :param mode: "w" or "r" or "rw" for write or read or readwrite
        """
        self._HeaderList=[]
        self._HeaderDict = {}
        self._HeaderPosDict= {}

        self.fname = fname

        self.mode=mode
        if not mode in ['r','w','rw']: raise Exception("Mode has to be r, w or rw")
        if not 'w' in mode and not os.path.isfile(self.fname):
            raise Exception("Read file "+self.fname+" does not exist")
        if 'r' in mode and os.path.isfile(self.fname): self._collectHeaders()


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
        if not name in self._HeaderList:
            print("Header " + name + " was not found on file " + self.fname)
            return None

        if not name in self._HeaderDict:
            with HAR_IO(self.fname,'r') as IOObj:
                self._HeaderDict[name] = Header.HeaderFromFile(name, self._HeaderPosDict[name], IOObj)
                assert(isinstance(self._HeaderDict[name], Header))

        if getDeepCopy:
            return self._HeaderDict[name].copy_header()
        else:
            return self._HeaderDict[name]

    def _collectHeaders(self):
        """
        Find all Header on a file. This is a private method and does not take any arguments
        """
        with HAR_IO(self.fname,'r') as IOObj:
            while True:
                pos, name = IOObj.nextHeader()
                if not name: break
                name=name.strip().upper()
                if name in self._HeaderList:
                    raise Exception('Multiple Headers with name ' + name +' on file ' + self.fname)
                self._HeaderList.append(name)
                self._HeaderPosDict[name]=pos

    def HeaderNames(self):
        """
        Can be used to obtain a list of all Headers on the file associated with the HAR object

        :return: list(str)
        """
        # :type: () -> list[str]
        return self._HeaderList

    def removeHeader(self,Header):
        """
        Unregisters a header from the HAR object. Note, that this will not affect the content on file
        but only the Headers associated with the object. To update the file :func:`write_HAR_File` has to be invoked

        :param Header: Header to be unregistered from teh HAR object
        :type Header: Header
        :return:
        """

        if not 'w' in self.mode: raise Exception("Cannot delete Header to HAR "+self.fname+" as it was declared mode r")

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

        if not 'w' in self.mode: raise Exception("Cannot add Header to HAR "+self.fname+" as it was declared mode r")


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
        if not 'w' in self.mode: raise Exception("Cannot write to file "+self.fname+" as it was declared mode r")

        for name in self._HeaderList:
            if not name in self._HeaderDict:
                self.getHeader(name)
        with HAR_IO(self.fname,'w') as IOObj:
            IOObj.truncate()
            for name in self._HeaderList:
                Header=self._HeaderDict[name]
                if Header.is_valid:
                    Header.HeaderToFile(IOObj)

    @staticmethod
    def cmbhar(inFileList,outfile):
        """


        :param inFileList: list of filenames whose content has to be combined
        :type: inFileList: List of strings
        :param outfile: output file name
        :type outfile: str
        :return:
        """
        HARobj=HAR(outfile,'w')
        harList=[]
        for myFile in inFileList:
            harList.append(HAR(myFile,'r'))
        refHAR=harList[0]
        for name in refHAR.HeaderNames():
            AllList=[refHAR.getHeader(name)]
            if not 'float' in str(AllList[0].DataObj.dtype):
                HARobj.addHeader(AllList[0])
            else:
                for hars in harList[1:]:
                    AllList.append(hars.getHeader(name))
                CmbedHeader=Header.concatenate(AllList,elemList=['File'+str(i) for i in range(0,len(inFileList))],
                                               headerName=name)
                HARobj.addHeader(CmbedHeader)
        return HARobj


    @staticmethod
    def diffhar(inFileList, outfile):
        """
        computes the running difference between a set of har files.
        Differences are always taken between consecutive entries in the inFileList

        :param inFileList: list of filenames whose content will be diffed
        :type: inFileList: List of strings
        :param outfile: output file name
        :type outfile: str
        :return:
        """
        HARobj = HAR(outfile, 'w')
        harList = []
        for myFile in inFileList:
            harList.append(HAR(myFile, 'r'))
        refHAR = harList[0]
        for name in refHAR.HeaderNames():
            AllList = [refHAR.getHeader(name)]
            if not 'float' in str(AllList[0].DataObj.dtype):
                HARobj.addHeader(AllList[0])
            else:
                for hars in harList[1:]:
                    AllList.append(hars.getHeader(name))
                elemList=['File' + str(i+1) +"-"+str(i) for i in range(0, len(inFileList)-1)]
                CmbedHeader = Header.runningDiff(AllList, elemList=elemList,headerName=name)
                HARobj.addHeader(CmbedHeader)
        return HARobj



