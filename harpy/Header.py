from __future__ import print_function, absolute_import
from .HeaderData import HeaderData
import numpy as np
import sys
import traceback
import itertools

__docformat__ = 'restructuredtext en'
genHeaderID = 0

class Header(HeaderData):
    DataType = ''

    def __init__(self, HeaderName=''):
        HeaderData.__init__(self)
        """

        :rtype: HeaderData
        """
        self.is_valid = True
        self.Error = ''
        self._HeaderName = HeaderName

    @classmethod
    def HeaderFromFile(cls, name, pos, HARFile):
        cls=Header()
        cls.f=HARFile
        cls._HeaderName=name

        try:
            cls._readHeader(pos)
        except:
            cls._invalidateHeader()
        return cls

    def _invalidateHeader(self):
        traceback.print_exc()
        print('Encountered an error in Header ' + self._HeaderName + ':\n')
        self.Error = sys.exc_info()[1]

    @classmethod
    def HeaderFromData(cls, name, array, label=None, CName=None,sets=None,SetElements=None):
        cls=Header()
        cls.HeaderName=name
        cls.DataObj=array
        if label: cls.HeaderLabel=label
        else: cls.HeaderLabel=name
        if CName: cls.CoeffName=CName
        else: cls.CoeffName=name
        if sets:
            if not SetElements: SetElements=[None for i in range(0,len(sets))]
        cls.SetNames=sets
        if sets:
            if isinstance(SetElements,list):
                if not isinstance(SetElements[0],list): raise Exception("Set Elements must be list of list of element names, received only list of element names")
                cls.SetElements=dict(zip(sets,SetElements))
            elif isinstance(SetElements,dict) : cls.SetElements= SetElements
            else: raise Exception("Set Elements have to be a list of lists (in set order) or a dict mapping sets to elements")

        return cls

    @classmethod
    def SetHeaderFromData(cls,name,array,SName,label=None):
        setLabel="Set "+SName
        if label:setLabel=setLabel+" "+label
        return cls.HeaderFromData(name, array, setLabel)


    def HeaderToFile(self,HARFile):
        self.f=HARFile
        pos=self.f.tell()
        try:
            self._writeHeader()
        except:
            traceback.print_exc()
            print('Error while writing Header "' + self._HeaderName + '"\n The program will continue but header is not put on file')
            self.f.seek(pos)
            self.f.truncate()


    @property
    def HeaderName(self):
        return self._HeaderName
    @HeaderName.setter
    def HeaderName(self, string4):
        if not isinstance(string4, str): raise Exception('Name not a string')
        if not len(string4) <= 4: raise Exception('Header name has to be shorter than 5')
        self._HeaderName = string4

    @property
    def HeaderLabel(self):
        return self._LongName
    @HeaderLabel.setter
    def HeaderLabel(self, string70):
        if not isinstance(string70, str): raise Exception('Name not a string')
        if not len(string70) <= 70: raise Exception('Header label has to be shorter than 70')
        self._LongName=string70.ljust(70)

    @property
    def CoeffName(self):
        return self._Cname
    @CoeffName.setter
    def CoeffName(self,string12):
        if not isinstance(string12, str): raise Exception('Name not a string')
        if not len(string12) <= 12: raise Exception('CoeffName has to be less then 12 characters')
        self._Cname = string12

    def getDataCopy(self):
        return np.ascontiguousarray(self._DataObj)

    @property
    def DataObj(self):
        return self._DataObj
    @DataObj.setter
    def DataObj(self, array):
        if not isinstance(array,np.ndarray):
            array=np.array(array,order='F')
        shape=array.shape
        dtype=str(array.dtype)
        if "|" in dtype and array.ndim>1:
            raise Exception("Version 1 Headers can only have scalar or vectorial string data")
        else:
            if "int" in dtype and array.ndim>2:
                raise Exception("Version 1 Headers can only have up to 2D string data")
            elif "float" in dtype and array.ndim > 7:
                raise Exception("Version 1 Headers can only have up to 2D string data")
            if "64" in dtype: raise Exception("Can only write 4byte data in Version 1 Headers")
        if self._setNames:
            if len(self._setNames) != array.ndim: raise Exception("Mismatch between data and set rank")
            for i,idim,els in enumerate(zip(array.shape, self._DimDesc)):
                if idim != len(els): raise Exception("Mismatch between data and set dimension "+str(i))
        self._DataObj=np.asfortranarray(array)

    @property
    def SetNames(self):
        return self._setNames
    @SetNames.setter
    def SetNames(self,names):
        if not names:
            self._setNames=None
            return
        if isinstance(names,str): names=[names]
        elif not isinstance(names,list): raise Exception('SetNames have to str or list of strings')
        if not all([isinstance(item,str) for item in names]):
            raise Exception('Found non string item in SetList, only strings can be set names')
        if not all([len(item)<=12 for item in names]):
            raise Exception('Maximum length for set Names is limited to 12 characters')

        if not self._DataObj is None:
            if self._DataObj.ndim==0 and names:
                raise Exception("Scalar can not have set elements associated")
            if self._DataObj.ndim < len(names):
                raise Exception("Number of sets higher than rank of Data array")
            elif self._DataObj.ndim > len(names):
                raise Exception("Number of sets lower than rank of Data array")
        self._setNames=names[:]

    @property
    def SetElements(self):
        return dict(zip(self._setNames, self._DimDesc))
    @SetElements.setter
    def SetElements(self,elDict):
        """
        Set elements from a dictionary {SetName : ElementList}
        Element list has to be either a list of strings max len 12 or None to indicate a numerical index
        Will fail if set is not in Header
        """
        if not isinstance(elDict,dict): raise Exception("Argument for SetElements needs to be dict")
        for key,val in elDict.items():
            if not any ([key.strip()== setn.strip() for setn in self._setNames]):
                raise Exception("Can not set Elements as set is not present in Header")
            indices=[i for i,set in enumerate(self._setNames) if key.strip() == set.strip()]
            if val:
                if not all([(isinstance(item, str)) for item in val]):
                    raise Exception("Element list must only contain strings")
                if not all([len(item) <= 12 for item in val]):
                    raise Exception('Maximum length for set Elements is limited to 12 characters')

            for i in indices:
                if not self._DimDesc: self._DimDesc=[None for j in range(0,len(self._setNames))]
                if not self._DimType: self._DimType=['NUM' for j in range(0,len(self._setNames))]
                if not val:
                    self._DimType[i]= "NUM"
                    self._DimDesc[i]= None
                elif self.DataObj.shape[i] != len(val):
                    raise Exception("Mismatch between number of elements and size of Data")
                else:
                    self._DimType[i] = "Set"
                    self._DimDesc[i]=val[:]

    def __getitem__(self, item):
        # type: (list) -> Header
        if len(item) != self._DataObj.ndim: raise Exception("Rank mismatch in indexing")
        if all([isinstance(i,int) for i in item]):
            ilist=[ [i] for i in item]
            return self._createDerivedHeader(ilist)

        if not all ([(isinstance(i,str,slice,int,list) for i in item)]):
            raise Exception("Index error Can only use int,str or slice as index")
        #TODO: maybe introduce a dict for the El ind mapping
        ilist=[]
        for ind,Els in zip(item,self._DimDesc):
            if isinstance(ind,slice):
                indList=[ind.start,ind.stop,ind.step]
                if all( [ i is None or isinstance(i,int) for i in indList] ):
                    ilist.append(ind)
                else:
                    if isinstance(ind.step,str): raise Exception("Elements not allowed as stride")
                    if isinstance(ind.start,str): start=Els.index(ind.start)
                    else: start=ind.start
                    if isinstance(ind.stop,str): stop=Els.index(ind.stop)+1
                    else: stop=ind.stop
                    ilist.append(slice(start,stop,ind.step))
            elif isinstance(ind,list):
                if not all([isinstance(i,(int,str)) for i in ind]):
                    raise Exception("Index list must only contain integer or str")
                ilist.append([i if isinstance(i,int) else Els.index(i) for i in ind])
            else:
                if isinstance(ind, str): start=Els.index(ind)
                else: start=ind
                #needed to keep the rank of the resulting matrix
                ilist.append(start)

        return self._createDerivedHeader(ilist)


    def _createDerivedHeader(self,indexList):

        label="Derivative of "+self.HeaderLabel
        if len(label) > 70: label = label[0:70]
        CName="Derived"
        sets=[]; SetElements=[]
        for i in range(0,len(indexList)):
            if not isinstance(indexList[i],int) : sets.append("S"+str(i))
            if self._DimDesc:
                if self._DimDesc[i]:
                    if isinstance(indexList[i],list):
                        SetElements.append([self._DimDesc[i][j] for j in indexList[i]])
                    elif isinstance(indexList[i],int):
                        pass
                    else:
                        SetElements.append(self._DimDesc[i][indexList[i]])
                else:
                    SetElements.append(None)
            else:
                SetElements=None
        print (sets,SetElements)
        array= self._DataObj[tuple(indexList)]
        print (array)


        return self.HeaderFromData(self.mkHeaderName(), array, label=label, CName=CName,
                                   sets=sets,SetElements=SetElements)



    def __sub__(self, other):
        if not isinstance(other,Header):
            raise Exception("Non Header object in subtraction")
        if not self.DataObj.shape == other.DataObj.shape:
            raise Exception("Can not subtract Headers with different shape")

        newarray=self.DataObj-other.DataObj
        return self.HeaderFromData(self.mkHeaderName(), newarray, label="Sub Result", CName="Subtract",
                                   sets=self.SetNames,SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def __add__(self, other):
        if not isinstance(other,Header):
            raise Exception("Non Header object in subtraction")
        if not self.DataObj.shape == other.DataObj.shape:
            raise Exception("Can not subtract Headers with different shape")

        newarray=self.DataObj+other.DataObj
        return self.HeaderFromData(self.mkHeaderName(), newarray, label="Sub Result", CName="Subtract",
                                   sets=self.SetNames,SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def append(self,other,axis):
        pass

    @classmethod
    def concatenate(cls,headerList,setName='',elemList=None,headerName=''):
        if not headerName: headerName=Header.mkHeaderName()
        if not setName: setName='CONCAT'
        if not elemList: elemList=['elem'+str(i) for i in range(0,len(headerList))]
        if len(elemList) != len(headerList):
            raise Exception("Size of element List does not match number of Headers in concatenation")
        refData=headerList[0].DataObj
        if not all([item.DataObj.shape == refData.shape for item in headerList]):
            raise Exception("Can not concatenate Headers with different shape")

        oldshape=list(refData.shape)
        oldshape.append(len(headerList))
        newarray=np.ndarray(tuple(oldshape),order='F',dtype=refData.dtype)
        for i in range(len(headerList)):
            newarray[...,i]=headerList[i].DataObj[...]

        newset=headerList[0].SetNames[:]
        newset.append(setName)
        newDesc=headerList[0]._DimDesc[:]
        newDesc.append(elemList)

        return Header.HeaderFromData(headerName, newarray, label="Concatenated", CName="Concat",
                                   sets=newset,SetElements=newDesc)

    @classmethod
    def runningDiff(cls,headerList,setName='',elemList=None,headerName=''):
        if not headerName: headerName=Header.mkHeaderName()
        if not setName: setName='CONCAT'
        if not elemList: elemList=['elem'+str(i) for i in range(0,len(headerList)-1)]
        if len(elemList) != len(headerList)-1:
            raise Exception("Size of element List does not match number of Headers in runningDiff")
        refData=headerList[0].DataObj
        if not all([item.DataObj.shape == refData.shape for item in headerList]):
            raise Exception("Can not take differences of Headers with different shape")

        oldshape=list(refData.shape)
        oldshape.append(len(headerList)-1)
        newarray=np.ndarray(tuple(oldshape),order='F',dtype=refData.dtype)
        for i in range(len(headerList)-1):
            newarray[...,i]=headerList[i+1].DataObj[...]-headerList[i].DataObj[...]

        newset=headerList[0].SetNames[:]
        newset.append(setName)
        newDesc=headerList[0]._DimDesc[:]
        newDesc.append(elemList)

        return Header.HeaderFromData(headerName, newarray, label="running Diffs", CName="RDiffs",
                                   sets=newset,SetElements=newDesc)


    @staticmethod
    def mkHeaderName():
        global genHeaderID
        name = (str(genHeaderID) + "_____")[0:4]
        genHeaderID+=1
        return name

    def toIndexList(self):
        ElementsSetList=[]
        SetElDict=self.SetElements
        for thisSet in self.SetNames:
            ElementsSetList.append(SetElDict[thisSet])

        flatDat=self.DataObj.flatten()
        return zip(itertools.product(*ElementsSetList),flatDat)












