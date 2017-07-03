from __future__ import print_function, absolute_import
from .HAR_IO import HAR_IO
from .Header import Header
from copy import deepcopy

__docformat__ = 'restructuredtext en'


class HAR(object):

    def __init__(self, fname, mode):
        # type: (str) -> HAR
        self._HeaderList=[]
        self._HeaderDict = {}
        self._HeaderPosDict= {}

        self.f = HAR_IO(fname,mode)
        self.fname = fname

        self.collectHeaders()

    def getHeader(self, name, getDeepCopy=False):
        # type: (str,bool) -> Header
        """

        :param name: str
        :return: Header
        :rtype: Header
        """
        if not name in self._HeaderList:
            print("Header " + name + " was not found on file " + self.fname)
            return None

        if not name in self._HeaderDict:
            self._HeaderDict[name] = Header.HeaderFromFile(name, self._HeaderPosDict[name], self.f)

        if getDeepCopy:
            return deepcopy(self._HeaderDict[name])
        else:
            return self._HeaderDict[name]

    def collectHeaders(self):
        # type: () ->
        self.f.seek(0)
        while True:
            pos, name = self.f.nextHeader()
            if not name: break
            if name in self._HeaderList:
                raise Exception('Multiple Headers with name ' + name +' on file ' + self.f.f._HeaderName)
            self._HeaderList.append(name)
            self._HeaderPosDict[name]=pos

    def HeaderNames(self):
        # :type: () -> list[str]
        return self._HeaderList

    def removeHeader(self,Header):
        if Header._HeaderName in self._HeaderDict:
            self._HeaderList.remove(Header._HeaderName)

    def addHeader(self,Header,overwrite=False):
        # type: (Header, bool) -> None

        if Header._HeaderName in self._HeaderDict and not overwrite:
            #print ("Header with name '" + Header._HeaderName + "' already on file")
            return
        else:
            if not Header._HeaderName in self._HeaderList:
                self._HeaderList.append(Header.HeaderName)
            self._HeaderDict[Header.HeaderName]=Header

    def write_HAR_File(self):
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

    @classmethod
    def cmbhar(cls,inFileList,outfile):
        cls=HAR(outfile,'w')
        harList=[]
        for myFile in inFileList:
            harList.append(HAR(myFile,'r'))
        refHAR=harList[0]
        for name in refHAR.HeaderNames():
            AllList=[refHAR.getHeader(name)]
            if not 'float' in str(AllList[0].DataObj.dtype):
                cls.addHeader(AllList[0])
            else:
                for hars in harList[1:]:
                    AllList.append(hars.getHeader(name))
                CmbedHeader=Header.concatenate(AllList,elemList=['File'+str(i) for i in range(0,len(inFileList))],
                                               headerName=name)
                cls.addHeader(CmbedHeader)
        return cls


    @classmethod
    def diffhar(cls, inFileList, outfile):
        cls = HAR(outfile, 'w')
        harList = []
        for myFile in inFileList:
            harList.append(HAR(myFile, 'r'))
        refHAR = harList[0]
        for name in refHAR.HeaderNames():
            AllList = [refHAR.getHeader(name)]
            if not 'float' in str(AllList[0].DataObj.dtype):
                cls.addHeader(AllList[0])
            else:
                for hars in harList[1:]:
                    AllList.append(hars.getHeader(name))
                elemList=['File' + str(i+1) +"-"+str(i) for i in range(0, len(inFileList)-1)]
                CmbedHeader = Header.runningDiff(AllList, elemList=elemList,headerName=name)
                cls.addHeader(CmbedHeader)
        return cls



