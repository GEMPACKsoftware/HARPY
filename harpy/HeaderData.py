from __future__ import print_function, absolute_import
from .HeaderCommonIO import *


class HeaderData(object):

    def __init__(self):
        self.f = None
        self.FileDims = None
        self._LongName = ' ' * 70
        self.StorageType = ' '
        self._setNames = None
        self._DimType = None
        self._DimDesc = None
        self._Cname = ' '
        self._DataObj = None
        self._role = ''
        self.DataDimensions = []
        self._HeaderName = ''
        self.VersionOnFile = 0
        self.hasElements=False

    def _readHeader(self, pos):
        self.f.seek(pos)

        newpos, name = self.f.nextHeader()
        if newpos != pos or self._HeaderName != name.strip(): raise Exception(
            "Header " + self._HeaderName + "not at indicated position")

        Version, DataType, self.StorageType, self._LongName, self.FileDims = self.f.parseSecondRec(name)

        if Version == 1:
            self._role="data"
            if DataType == '1C':
                readHeader1C(self)
                if self._LongName.lower().startswith('set '):
                    self._role = 'set'
                    self._setNames= [self._LongName.split()[1]]
            elif DataType == 'RE':
                self.hasElements=True
                readHeader7D(self, True)
            elif DataType == 'RL':
                readHeader7D(self, False)
            elif DataType == '2R':
                readHeader2D(self, 'f')
            elif DataType == '2I':
                readHeader2D(self, 'i')



    def _writeHeader(self):

        typeString = str(self._DataObj.dtype)
        hasElements = isinstance(self._setNames,list)
        if 'float32' == typeString and (self._DataObj.ndim != 2 or hasElements):
            writeHeader7D(self)
        elif 'int32' == typeString or 'float32' == typeString:
            writeHeader2D(self)
        elif '<U' in typeString or '|S' in typeString:
            if self._DataObj.ndim > 1:
                print('"' + self.name + '" can not be written as Charcter arrays ndim>1 are not yet supported')
                return
            writeHeader1C(self)
        else:
            raise Exception('Can not write data in Header: "' +
                            self.name + '" as data style does not match any known Header type')

    def __str__(self):
        outList=[]
        outList.append("HeaderName:".ljust(20)+self._HeaderName)

        if 'float' in str(self._DataObj.dtype) : outList.append("\nDataType:".ljust(21) + "Real")
        elif 'int' in str(self._DataObj.dtype) : outList.append("\nDataType:".ljust(21) + "Integer")
        if '|' in str(self._DataObj.dtype) : outList.append("\nDataType:".ljust(21) + "String")


        if self._LongName:
            outList.append("\nDescription:".ljust(21) + self._LongName)
        if self._Cname:
            outList.append("\nCoefficientName:".ljust(21) + self._Cname)

        outList.append("\nRank:".ljust(21) + str(self._DataObj.ndim))
        outList.append("\nDimensions:".ljust(21) +' '.join([str(i) for i in self._DataObj.shape]))
        if self._setNames:
            outList.append("\nAssociatedSets:".ljust(21) +''.join([i.ljust(16) for i in self._setNames]))
            outList.append("\n")
            for set,type,els in zip(self._setNames, self._DimType, self._DimDesc):
                if type=='NUM':
                    outList.append("\n"+(set.strip()+":").ljust(21)+"Numeric set defined by its size")
                else:
                    outList.append("\n"+(set.strip()+":").ljust(20)+
                        ' '.join([els[i].ljust(15) if (i+1)%6 !=0 else "\n".ljust(21)+els[i].ljust(15) for i in range(0,len(els))]))
        else:
            outList.append("\n\nNo Sets associated with this Header")

        return ''.join(outList)