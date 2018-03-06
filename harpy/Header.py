from __future__ import print_function, absolute_import, division
from .HeaderData import HeaderData
from .HAR_IO import HAR_IO
from .HeaderCommonIO import readHeader1C, readHeader7D, readHeader2D
# ^^ Including them explicitly makes it easier to identify where they've come from
import numpy as np
import sys
import traceback
import itertools
import copy

__docformat__ = 'restructuredtext en'
genHeaderID = 0

class Header(HeaderData):
    """
    Contains all methods to manipulate data associate with the Header. This can:

    the Header name
    the Coefficient name
    the set names associated with its dimensions
    the elements of the sets
    the labelling information
    the data entries in the Header

    Access to these is provided by properties

    Header objects permit operation such as indexing and basic mathematical operations, +,-,*,/

    """

    if sys.version_info < (3,):
        as_unicode = False
    else:
        as_unicode = True


    def __init__(self, HeaderName='', f=None):
        """
        Inherits from HeaderData
        :rtype: Header
        """
        HeaderData.__init__(self) # Does not execute any HeaderData methods
        self.is_valid = True
        self.DataType=''
        self.Error = ''
        self.f = f # type: HAR_IO
        self._HeaderName = HeaderName

    @staticmethod
    def HeaderFromFile(name, HARFile):
        # type: (str, HAR_IO) -> Header
        """
        Reads a Header from file.
        This function should only be invoked from class HAR as this knows the position of the Header

        :param name: Name of the Header
        :param pos: position on the file
        :param (HAR_IO) HARFile: file object containing the Header
        :return: Header object
        """
        ret = Header(HeaderName=name, f=HARFile)
        assert isinstance(ret, Header)  # Something is seriously wrong if this raises...

        try:
            pos = ret.f.getHeaderPos(name)
            print("Header '%s' position is: %d" %(name, pos))
            ret._readHeader(pos)  # Method inherited from HeaderData class
        except:
            ret._invalidateHeader()

        assert isinstance(ret, Header)  # Something screwed up reading the header...
        return ret

    def _readHeaderByName(self, header_name):

        try:
            assert isinstance(self.f, HAR_IO)
        except AssertionError:
            raise TypeError("Header object has not been provided with type 'HAR_IO' object for attribute f.")

        (self._DataObj, self.StorageType, self._LongName, \
        self.FileDims, self.hasElements, self._role, self._setNames) = self.f.readHeaderByName(header_name,
                                                                                               parent_header=self)


    def _readHeader(self, pos):
        # type: (int) -> None

        try:
            assert isinstance(self.f, HAR_IO)
        except AssertionError:
            raise TypeError("Header object has not been provided with type 'HAR_IO' object for attribute f.")

        self.f.seek(pos)
        print(self.f.tell())

        newpos, name = self.f.nextHeader() # This relies on a subclass creating the 'f' object, which declares this method
        if newpos != pos or self._HeaderName != name.strip():
            raise RuntimeError("Header " + self._HeaderName + "not at indicated position")

        print("Header._readHeader() newpos, name ", newpos, name)

        Version, DataType, self.StorageType, self._LongName, self.FileDims = self.f.parseSecondRec(name)

        # readHeader methods alter self._DataObj, self.RealDim, self.DataDimension, self.StorageType possibly self.f
        if Version == 1:
            self._role="data"
            if DataType == '1C':
                readHeader1C(self)
                if self._LongName.lower().startswith('set '):
                    self._role = 'set'
                    self._setNames= [self._LongName.split()[1]]
            elif DataType == 'RE':
                self.hasElements=True
                print("Header._readHeader() RE self.f.tell() ", self.f.tell())
                readHeader7D(self, True)
            elif DataType == 'RL':
                readHeader7D(self, False)
            elif DataType == '2R':
                readHeader2D(self, 'f')
            elif DataType == '2I':
                readHeader2D(self, 'i')

        assert isinstance(self, Header)

    def _invalidateHeader(self):
        traceback.print_exc()
        print('Encountered an error in Header ' + self._HeaderName + ':\n')
        self.Error = sys.exc_info()[1]

    @classmethod
    def HeaderFromConstant(cls, name, label=None, CName=None,sets=None,SetElements=None, dims=None, value=None):
        if SetElements is None and dims is None:
            raise Exception("Header from constant needs either SetElements or dims")
        if not (SetElements is None or dims is None):
            raise Exception("Header from constant can only have SetElements or dims but not both")
        if not isinstance(value,(int,float)):
            raise Exception("value must be integer float")
        if value is None: value=0.0
        if not dims is None:
            if not isinstance(dims,tuple): raise("dims must be given as tuple (empty tuple for scalars)")
            array=np.full(dims,value)
        else:
            if not isinstance(SetElements,(list,dict)): raise Exception("SetElements must be list or dictionary")
            if isinstance(SetElements,list):
                dims=tuple([len(sublist) for sublist in SetElements])
            else:
                if not all([set in SetElements for set in sets]): raise Exception("Not all sets in SetElement Dict")
                dims=tuple([len(SetElements[set]) for set in sets])

        return cls.HeaderFromData(name, np.full(dims,value), label=label, CName=CName,sets=sets,SetElements=SetElements)

    @classmethod
    def HeaderFromData(cls, name, array, label=None, CName=None,sets=None,SetElements=None):
        """
        Creates a new Header object from basic data. only Header name and data array are mandatory

        :param str name: Header name (max 4 characters)
        :param numpy.ndarray array: data array contianing all values
        :param str label: long description of content
        :param str CName: Coefficicient name from which the Header is derived in TAB files
        :param list(str) sets: Name of the sets corresponding to each dimensions (size needs to be rank array)
        :param SetElements: list of list of elements (one per dim) or dict(setnames,elements)
        :type SetElements: list(list(str)) || dict(str:list(str))
        :return: new Header object
        :rtype: Header

        """
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
                if not (isinstance(SetElements[0],list) or SetElements[0] is None):
                    raise Exception("Set Elements must be list of list of element names, received only list of element names")
                cls.SetElements=dict(zip(sets,SetElements))
            elif isinstance(SetElements,dict) : cls.SetElements= SetElements
            else: raise Exception("Set Elements have to be a list of lists (in set order) or a dict mapping sets to elements")

        return cls

    @classmethod
    def SetHeaderFromData(cls,name,array,SName,label=None):
        """
        Create a Header object which contains the marker of a set Header (ViewHar might need to know)

        :param str name: Header Name
        :param numpy,chararray array: numpy array with the set elements
        :param str SName: Name of the set
        :param label: long description of the content of the sets
        :return: Header object which is recognized by Viewhar as Set
        :rtype: Header
        """
        setLabel="Set "+SName
        if label:setLabel=setLabel+" "+label
        myHeader=cls.HeaderFromData(name, array, setLabel)
        myHeader._role="set"
        return myHeader


    def HeaderToFile(self,HARIOInst):
        """
        Should only be called from class HAR

        :param HARIOInst: HAR_IO instance that the Header will be written
        :return:
        """
        self.f=HARIOInst
        pos=self.f.tell()
        try:
            self._writeHeader()
        except:
            traceback.print_exc()
            print('Error while writing Header "' + self._HeaderName + '"\n The program will continue but header is not put on file')
            self.f.seek(pos)
            self.f.truncate()

    def setProperties(self,HeaderName=None,Sets=None,CoeffName=None,DataObj=None, SetEl=None):
        """
        Utility function to set multiple properties of an exisiting Header at once
        Note, this can not be used to change dimensions and all variables are supposed to
        Conform the structure of the existing object

        :param str HeaderName:  Name of the Header
        :param list(str) Sets: Name of the sets
        :param str CoeffName: Coefficient name
        :param np.ndarray DataObj: the data associated with the object
        :param dict(str:list(str)) SetEl: Can be used to change a subset by omitting other entries in dictionary
        :return : None
        """
        if HeaderName:self.HeaderName=HeaderName
        if Sets:self.SetNames=Sets
        if CoeffName:self.CoeffName=CoeffName
        if DataObj:self.DataObj=DataObj
        if SetEl:
            if not isinstance(SetEl,dict): raise Exception
            tmpSetEl=self.SetElements
            for keys in SetEl.keys():
                tmpSetEl[keys]=SetEl[keys]
            self.SetElements=SetEl


    @property
    def HeaderName(self):
        """
        Property to retrive or set a pointer from/to the Headername (str max 4 characters)
        """
        return self._HeaderName
    @HeaderName.setter
    def HeaderName(self, string4):
        if not isinstance(string4, str): raise Exception('Name not a string')
        if not len(string4) <= 4: raise Exception('Header name has to be shorter than 5. Name is '+string4)
        self._HeaderName = string4.strip()

    @property
    def HeaderLabel(self):
        """
        Property to retrive or set a pointer from/to the Header descirption (str max 70 characters)
        :returns: Label of the Header
        :rtype: str
        """
        return self._LongName
    @HeaderLabel.setter
    def HeaderLabel(self, string70):
        if not isinstance(string70, str): raise Exception('Name not a string')
        if self._role=="set":
            if not string70.lower().startswith("set"):
                string70=" ".join(self.HeaderLabel.split()[0:2])+" "+string70
        if not len(string70) <= 70:
            #print ('Warning, Header Label too long, only using70 chars')
            string70=string70[0:70]
        self._LongName=string70.ljust(70)

    @property
    def CoeffName(self):
        """
        Property to retrive or set a pointer from/to the associated coefficient (str max 12 characters)
        """
        return self._Cname
    @CoeffName.setter
    def CoeffName(self,string12):
        if not isinstance(string12, str): raise Exception('Name not a string')
        if not len(string12) <= 12: raise Exception('CoeffName has to be less then 12 characters')
        self._Cname = string12

    @property
    def DataObj(self):
        """
        Property to retrive or set a pointer from/to the data array (str max 4 characters)
        """
        return self._DataObj
    @DataObj.setter
    def DataObj(self, array):
        if not isinstance(array,np.ndarray):
            array=np.array(array)
        shape=array.shape
        dtype=str(array.dtype)
        if "|" in dtype and array.ndim>1:
            raise Exception("Version 1 Headers can only have scalar or vectorial string data")
        else:
            if "int" in dtype and array.ndim>2:
                raise Exception("Version 1 Headers can only have up to 2D integer data")
            elif "float" in dtype and array.ndim > 7:
                raise Exception("Version 1 Headers can only have up to 7D Real data")
            #if "64" in dtype: raise Exception("Can only write 4byte data in Version 1 Headers")
        if self._setNames:
            if len(self._setNames) != array.ndim: raise Exception("Mismatch between data and set rank")
            for i,idim,els in enumerate(zip(array.shape, self._DimDesc)):
                if idim != len(els): raise Exception("Mismatch between data and set dimension "+str(i))
        self._DataObj=array.astype(dtype=dtype.replace("64","32"))

    @property
    def SetNames(self):
        """
        Property to retrive or set a pointer from/to the Set Names associated with the dimensions (list(str) with len of rank array)
        :rtype: list(str)
        """
        if self._role=="set":
            return [self.HeaderLabel.split()[1]]
        return self._setNames
    @SetNames.setter
    def SetNames(self,names):
        if names is None:
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
        """
        Property to retrive or set a pointer from/to the Set Elements associated with the dimensions
        The getter returns a dictionary of the form (set:[Elelement List])
        The setter expects a dict of the same form. If no elements are associated with a dimension None can be passed
        instead of a list.
        If it was a set definition it obtains the information from the stored data
        """
        if self._role=="set":
            return {self.HeaderLabel.split()[1] : [str(name.strip()) for name in self.DataObj]}
        return dict(zip(self._setNames, self._DimDesc))
    @SetElements.setter
    def SetElements(self,elDict):
        """
        Set elements from a dictionary {SetName : ElementList}
        Element list has to be either a list of strings max len 12 or None to indicate a numerical index
        Will fail if set is not in Header
        """
        if not isinstance(elDict,dict): raise Exception("Argument for SetElements needs to be dict")
        if not all([setn in elDict for setn in self._setNames]):
            raise Exception("Can not set Elements as set is not present in Header")
        for key,val in elDict.items():
            if not key.strip() in  self._setNames : continue
                #raise Exception("Can not set Elements as set is not present in Header")
            indices=[i for i,set in enumerate(self._setNames) if key.strip() == set.strip()]
            if val:
                if not all([(isinstance(item, str)) for item in val]):
                    print ([type(item) for item in val if not isinstance(item,str)])
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
        """
        Allows index access to Header objects. indicies can be numeric or element strings
        E.g. Head["Elem1",4] is valid.
        Slices, strides, index lists and ellipsis are implemented.
        If a single element is indexed, a Header with reduced dimensionality is returned
        E.g. Head[2,2,2] will return a scalar Header
        to retain a dimension in this acces use list access
        E.g. Head[2,2,[2]] will return a 1-D Header with 3rd dimension (3rd element) retianed

        :param item: indexing pattern
        :return: A new Header object
        """
        if not isinstance(item, tuple): item = [item]
        if len(item) != self._DataObj.ndim: raise Exception("Rank mismatch in indexing")
        if all([isinstance(i,int) for i in item]):
            ilist=[ [i] for i in item]
            return self._createDerivedHeader(ilist)

        if not all ([(isinstance(i,str,slice,int,list) for i in item)]):
            raise Exception("Index error Can only use int,str or slice as index")
        #TODO: maybe introduce a dict for the El ind mapping
        ilist = self.itemList_to_intIndex(item)

        return self._createDerivedHeader(ilist)

    def __setitem__(self, item, value):
        if not isinstance(item,tuple): item=[item]
        if len(item) != self._DataObj.ndim: raise Exception("Rank mismatch in indexing")
        if all([isinstance(i,int) for i in item]):
            ilist=[ [i] for i in item]
        elif not all ([(isinstance(i,str,slice,int,list) for i in item)]):
            raise Exception("Index error Can only use int,str or slice as index")
        else:
            ilist = self.itemList_to_intIndex(item)

        newinds, shape = self.makeNPIndex(ilist)
        if isinstance(value,Header):
            self._DataObj[newinds]=value.DataObj
        elif isinstance(value,(int,np.ndarray,float)):
            self._DataObj[newinds] = value
        else:
            raise Exception("Can only set Header via Header, numpy object or number")

    def itemList_to_intIndex(self, item):
        ilist = []
        for ind, Els in zip(item, self._DimDesc):
            if isinstance(ind, slice):
                indList = [ind.start, ind.stop, ind.step]
                if all([i is None or isinstance(i, int) for i in indList]):
                    ilist.append(ind)
                else:
                    if isinstance(ind.step, str): raise Exception("Elements not allowed as stride")
                    if isinstance(ind.start, str):
                        start = Els.index(ind.start)
                    else:
                        start = ind.start
                    if isinstance(ind.stop, str):
                        stop = Els.index(ind.stop) + 1
                    else:
                        stop = ind.stop
                    ilist.append(slice(start, stop, ind.step))
            elif isinstance(ind, list):
                if not all([isinstance(i, (int, str)) for i in ind]):
                    raise Exception("Index list must only contain integer or str")
                ilist.append([i if isinstance(i, int) else Els.index(i.strip()) for i in ind])
            else:
                if isinstance(ind, str):
                    start = Els.index(ind)
                else:
                    start = ind
                # needed to keep the rank of the resulting matrix
                ilist.append(start)
        return ilist


    def _createDerivedHeader(self,indexList):

        label="Derivative of "+self.HeaderLabel
        if len(label) > 70: label = label[0:70]
        CName="Derived"
        sets=[]; SetElements=[]
        for i in range(0,len(indexList)):
            #Create New set names if necessary
            if not isinstance(indexList[i],int) :
                if isinstance(indexList[i],slice):
                    if indexList[i].start is None and indexList[i].stop is None and indexList[i].step is None:
                        sets.append(self.SetNames[i])
                    else:
                        sets.append("S" + str(i))
                else:
                    sets.append("S" + str(i))
            #Puick the elements
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
# Deal with advanced indexing
        newinds, shape = self.makeNPIndex(indexList)
        if shape:
            array= np.reshape(self._DataObj[newinds],tuple(shape))
        else:
            array=self._DataObj[newinds]

        return self.HeaderFromData(self._mkHeaderName(), array, label=label, CName=CName,
                                   sets=sets, SetElements=SetElements)

    def makeNPIndex(self, indexList):
        if any([isinstance(item, list) for item in indexList]):
            newinds = []; shape=[]
            for i, item in enumerate(indexList):
                if isinstance(item, int):
                    newinds.append([item])
                elif isinstance(item, slice):
                    tmp = [i for i in range(0, self._DataObj.shape[i])]
                    newinds.append(tmp[item])
                    shape.append(len(newinds[-1]))
                elif isinstance(item, list):
                    newinds.append(item)
                    shape.append(len(newinds[-1]))
            numpyInd = np.ix_(*newinds)
        else:
            shape = None
            numpyInd = tuple(indexList)

        return numpyInd, shape

    def __rsub__(self, other):
        op="Subtraction"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=other.DataObj-self.DataObj
        elif isinstance(other,(int,float)):
            newarray=other-self.DataObj
        else:
            raise Exception("Only Header or scalar allowed in subtraction")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Subtract",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def __sub__(self, other):
        op="Subtraction"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=self.DataObj-other.DataObj
        elif isinstance(other,(int,float)):
            newarray=self.DataObj-other
        else:
            raise Exception("Only Header or scalar allowed in subtraction")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Subtract",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def __radd__(self, other):
        return self+other

    def __add__(self, other):
        op="Addition"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=self.DataObj+other.DataObj
        elif isinstance(other, np.ndarray):
            newarray = self.DataObj + other
        elif isinstance(other,(int,float)):
            newarray=self.DataObj+other
        else:
            raise Exception("Only Header or scalar allowed in addition")

        SetEls=None
        if not self.SetNames is None:
            SetEls=[self.SetElements[nam] for nam in self.SetNames]

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Add",
                                   sets=self.SetNames, SetElements=SetEls)

    def __rmul__(self, other):
        return self*other

    def __mul__(self, other):
        op="Multiplication"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=self.DataObj*other.DataObj
        elif isinstance(other, np.ndarray):
            newarray = self.DataObj * other
        elif isinstance(other,(int,float)):
            newarray=self.DataObj*other
        else:
            raise Exception("Only Header or scalar allowed in multiplication")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Multiply",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])
    def __rfloordiv__(self, other):
        op="Division"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=other.DataObj//self.DataObj
        elif isinstance(other,np.ndarray):
            newarray = other // self.DataObj
        elif isinstance(other,(int,float)):
            newarray=other//self.DataObj
        else:
            raise Exception("Only Header or scalar allowed in division")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Divide",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def __floordiv__(self, other):
        op="Division"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=self.DataObj//other.DataObj
        elif isinstance(other, np.ndarray):
            newarray=self.DataObj//other
        elif isinstance(other,(int,float)):
            newarray=self.DataObj//other
        else:
            raise Exception("Only Header or scalar allowed in division")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Divide",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def __rtruediv__(self, other):
        op="Division"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=other.DataObj/self.DataObj
        elif isinstance(other,np.ndarray):
            newarray = other / self.DataObj
        elif isinstance(other,(int,float)):
            newarray=other/self.DataObj
        else:
            raise Exception("Only Header or scalar allowed in division")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Divide",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def __truediv__(self, other):
        op="TrueDivision"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=self.DataObj/other.DataObj
        elif isinstance(other, np.ndarray):
            newarray=self.DataObj/other
        elif isinstance(other,(int,float)):
            newarray=self.DataObj/other
        else:
            raise Exception("Only Header or scalar allowed in division")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Divide",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])

    def __pow__(self,other):
        op="Power"
        if isinstance(other,Header):
            self._verifyHeaders(op, other)
            newarray=self.DataObj**other.DataObj
        elif isinstance(other, np.ndarray):
            newarray = self.DataObj ** other
        elif isinstance(other,(int,float)):
            newarray=self.DataObj**other
        else:
            raise Exception("Only Header or scalar allowed in Power")

        return self.HeaderFromData(self._mkHeaderName(), newarray, label="Sub Result", CName="Power",
                                   sets=self.SetNames, SetElements=[self.SetElements[nam] for nam in self.SetNames])


    def _verifyHeaders(self, op, other):
        if not self.DataObj.shape == other.DataObj.shape:
            raise Exception("Headers with different shape not permitted in " + op)
        # if self.SetNames != other.SetNames:
        #     print("Warning: Headers " + self.HeaderName + " and " + other.HeaderName + " in " + op + " have different sets associated.")
        #     print (self.SetElements)
        #     print (other.SetElements)
        #     print("         Operation will proceed but make sure to investigate")
        #     if not all([self.SetElements[i] == other.SetElements[i] for i in self.SetNames]):
        #         print("Warning: Set elements in Headers " + self.HeaderName + " and " + other.HeaderName + " in " + op + " do not match")
        #         print("         Operation will proceed but make sure to investigate")


    def append(self,other,axis):
        pass

    @classmethod
    def concatenate(cls,headerList,setName='',elemList=None,headerName=''):
        if not headerName: headerName=Header._mkHeaderName()
        if not setName: setName='CONCAT'
        if not elemList: elemList=['elem'+str(i) for i in range(0,len(headerList))]
        if len(elemList) != len(headerList):
            raise Exception("Size of element List does not match number of Headers in concatenation")
        refData=headerList[0].DataObj
        if not all([item.DataObj.shape == refData.shape for item in headerList]):
            raise Exception("Can not concatenate Headers with different shape")

        oldshape=list(refData.shape)
        oldshape.append(len(headerList))
        newarray=np.ndarray(tuple(oldshape),dtype=refData.dtype)
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
        if not headerName: headerName=Header._mkHeaderName()
        if not setName: setName='CONCAT'
        if not elemList: elemList=['elem'+str(i) for i in range(0,len(headerList)-1)]
        if len(elemList) != len(headerList)-1:
            raise Exception("Size of element List does not match number of Headers in runningDiff")
        refData=headerList[0].DataObj
        if not all([item.DataObj.shape == refData.shape for item in headerList]):
            raise Exception("Can not take differences of Headers with different shape")

        oldshape=list(refData.shape)
        oldshape.append(len(headerList)-1)
        newarray=np.ndarray(tuple(oldshape),dtype=refData.dtype)
        for i in range(len(headerList)-1):
            newarray[...,i]=headerList[i+1].DataObj[...]-headerList[i].DataObj[...]

        newset=headerList[0].SetNames[:]
        newset.append(setName)
        newDesc=headerList[0]._DimDesc[:]
        newDesc.append(elemList)

        return Header.HeaderFromData(headerName, newarray, label="running Diffs", CName="RDiffs",
                                   sets=newset,SetElements=newDesc)


    @staticmethod
    def _mkHeaderName():
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

    def nan_to_number(self, number=0):
        self._DataObj=np.nan_to_num(self._DataObj,number)

    @classmethod
    def sum(cls,Header,axis=None):
        outHeader=cls._reduce(Header,axis=axis,npfunction=np.sum)
        return outHeader

    @classmethod
    def mean(cls,Header,axis=None):
        outHeader=cls._reduce(Header,axis=axis,npfunction=np.mean)
        return outHeader

    @classmethod
    def _reduce(cls,Header,axis=None,npfunction=None):
        if not axis is None:
            if isinstance(axis,str):
                myaxis = Header.setNameToIndex(axis)
            elif isinstance(axis,int):
                if axis > len(Header._setNames) -1 : raise  Exception("Axis out of range")
                myaxis=axis
            elif isinstance(axis,list):
                if any( [ not isinstance(item,int) for item in list] ) :
                    raise Exception("sum with axis=list requires all elements to be integers ")
                if any( [ item > len(Header._setNames) -1 for item in list] ) :
                    raise Exception("item in list out of range of dimensions")
            else:
                raise Exception("wrong type for axis argument")

            newarray=npfunction(Header.DataObj,axis=myaxis)
            newSet=Header.SetNames[0:myaxis]+Header._setNames[myaxis+1:]
        else:
            newarray = npfunction(Header.DataObj)
            newSet=[]
        return cls.HeaderFromData(cls._mkHeaderName(),array=newarray,sets=newSet,SetElements=Header.SetElements)

    def setNameToIndex(self, axis):
        nsets = self._setNames.count(axis)
        if nsets == 0: raise Exception(
            "Setname " + axis + " is invalid. Available Sets are:" + ",".join(self._setNames))
        if nsets > 1: raise Exception(
            "Setname " + axis + " appears in more than one dimensions. Need to use int instead")
        myaxis = self._setNames.index(axis)
        return myaxis

    def copy_header(self):
        new = Header(HeaderName=self._HeaderName)

        attrs = set(self.__dict__.keys())
        attrs.remove('f') # Remove unserialisable attributes

        for k in attrs:
            setattr(new, k, copy.deepcopy(getattr(self, k)))

        return new
