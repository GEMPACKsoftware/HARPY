"""
Created on Jun 29 14:46:48 2018

"""

import numpy as np
from typing import List, Union

class _HeaderSet:
    """
    This class is used to represent sets associated with header arrays.
    """

#    Status is unknown elements but set, element index, known set elements, no set just numeric
    _valid_status = ["u", "e", "k", "n"]
    _genSetID     = 0

    def __init__(self, name: 'Union[str,None]',
                 status: str,
                 dim_desc: 'Union[List[str],str,None]',
                 dim_size: int):

        self.name = name
        self.status = status
        self.dim_desc = dim_desc
        if not dim_desc is None:
            if any([len(el) > 12 for el in dim_desc]):
                raise ValueError("Set Element too long (maximum 12 Characters for set Elements)")
        self.elemPosDict={} if self.dim_desc is None else dict(zip( [elem.strip().lower() for elem in dim_desc], range(0,len(self.dim_desc))))
        self.dim_size = dim_size

    def transform_index(self,index):
        if isinstance(index,(str,int)):
            return self.name_to_ind(index), None

        elif isinstance(index,slice):
            newslice=self.convertSlice(index)
            npIndList=list(range(self.dim_size))[newslice]
            SetName=self._newname() if not all(p is None for p in [newslice.start,newslice.stop,newslice.step]) else self.name
            if self.dim_desc:
                return npIndList, _HeaderSet(SetName, self.status, self.dim_desc[newslice], len(npIndList))
            else:
                return npIndList, _HeaderSet(SetName, self.status, dim_desc=None, dim_size=len(npIndList))


        elif isinstance(index,list):
            useElem=self.status in ["e","k"]
            setElList=[] if useElem else None
            npIndList=[]
            for ind in index:
                if isinstance(ind, (str,int) ):
                    idx=self.name_to_ind(ind)
                    npIndList.append(idx)
                    if useElem: setElList.append(self.dim_desc[idx])
                elif isinstance(ind,slice):
                    newslice = self.convertSlice(ind)
                    npIndList.append(list(range(self.dim_size))[newslice])
                    if useElem: setElList.extend(self.dim_desc[newslice])
                else:
                    raise TypeError("Only slice, str, int allowed in list indexing")
            if useElem:
                if len(set(setElList)) != len(setElList):
                    raise ValueError("Indexing leads to duplicate set elements which is not permitted")
                if setElList != self.dim_desc:
                    return npIndList, _HeaderSet(self._newname(), self.status, setElList, len(npIndList))
                else:
                    return npIndList, self
            else:
                return npIndList, _HeaderSet(self._newname(), self.status, None, len(npIndList))


    def convertSlice(self,index):
        if not isinstance(index.step, int) and not index.step is None:
            raise ValueError("step in slice has to be integer")
        start=self.name_to_ind(index.start)
        start= None if start==0 else start
        stop = self.name_to_ind(index.stop)
        stop =  None if stop==self.dim_size else stop
        step= None if index.step == 1 else index.step
        return slice(start, stop, step)

    def name_to_ind(self,idx):
        if idx is None:
            return None
        elif isinstance(idx,str):
            if idx.strip().lower() in self.elemPosDict:
                return self.elemPosDict[idx.strip().lower()]
            else:
                raise ValueError("Element not in set")
        elif isinstance(idx,int):
            if idx >= self.dim_size:
                raise ValueError("Index Out Of bounds")
            return idx

    def _newname(self):
        _HeaderSet._genSetID+=1
        return "S@"+str(_HeaderSet._genSetID)


class _HeaderDims:

    def __init__(self, setList):
        self._dims=setList

    @staticmethod
    def fromShape(shape):
        setList=[_HeaderSet(None, 'n', None, dim) for dim in shape]
        return _HeaderDims(setList)

    @staticmethod
    def fromSetShape(sets, setElDict, shape):
        setObjList=[]
        lowerDict=dict(zip([key.strip().lower() for key in setElDict.keys()], setElDict.keys() ))
        for idim, setName in enumerate(sets):
            lowSet=setName.strip().lower()
            if lowSet in lowerDict:
                setObjList.append(_HeaderSet(setName,'k',setElDict[lowerDict[lowSet]],shape[idim]))
            else:
                setObjList.append(_HeaderSet(setName, 'u', None, shape[idim]))
        return _HeaderDims(setObjList)


    @property
    def dims(self) -> List[_HeaderSet]:
        return self._dims

    @dims.setter
    def dims(self, obj) -> None:
        self._dims = obj

    def ndim(self):
        """
        Number of dimensions
        """
        return len(self._dims)

    def defined(self):
        """
        Tells whether dimensensions have sets defined or are just array like
        """
        return not all([dim.name is None for dim in self._dims])

    @property
    def setNames(self):
        return [dim.name for dim in self.dims]
    
    @setNames.setter
    def setNames(self, sNames):
        if not isinstance(sNames,list): raise TypeError("set Names needs to be given as a list of strings")
        if len(sNames) != len(self.dims) : raise ValueError("wrong length of set List. Header is rank "+str(len(self.dims))+ "but received list size "+ len(sNames))
        for name in sNames:
            if not isinstance(name,str): raise TypeError("set Names contains a non string object: "+str(name))
            if len(name.strip()) > 12 : raise ValueError("Set names are limited to 12 characters. received '"+name+"'")
        for newName, dim in zip(sNames,self.dims):
            dim.name=newName.strip()

    @property
    def setElements(self):
        return [dim.dim_desc for dim in self.dims]

    @property
    def shape(self):
        return tuple([sets.dim_size for sets in self._dims])

    def __str__(self):
        outputstr=""
        for setDim in self._dims:
            if setDim.status in "keu":
                outputstr+="   " + setDim.name.ljust(12) + ": \n"
            else:
                outputstr+="   "+"Not Specified"
            if setDim.status in "ke":
                outputstr+="      " +", ".join(setDim.dim_desc) + "\n"
        return outputstr



    def compatible_shape(self,other):
        return self.shape == other

    def matchSets(self,sets=None, shape:tuple=None):
        if sets is None and shape is None : raise KeyError("Only one argument allowed")
        newSets = []
        if not sets is None:
            # Try to match the shape of the dimensions
            iset=len(self.dims)-1; jset=len(sets.dims)-1
            while iset >=0 and jset >=0:
                if jset < 0 :
                    newSets.append(self.dims[iset])
                    iset -=1
                elif iset < 0 :
                    newSets.append(sets.dims[jset])
                    jset -=1
                if self.dims[iset].dim_size == sets.dims[jset].dim_size or self.dims[iset].dim_size == 1 or sets.dims[jset].dim_size == 1:
                    if self.dims[iset].status != 'n':
                        newSets.append(self.dims[iset])
                    else:
                        newSets.append(sets.dims[jset])
                    iset-= 1 ; jset -=1
            newSets.reverse()
        elif not shape is None:
            iset = len(self.dims) - 1; jset=len(shape)-1
            while iset >=0 and jset >=0:
                if jset < 0 :
                    newSets.append(self.dims[iset])
                    iset -=1
                elif iset < 0 :
                    newSets.append(_HeaderSet(None , 'n' , None, shape[jset]))
                    jset -=1
                if self.dims[iset].dim_size == shape[jset] or self.dims[iset].dim_size == 1 or shape[jset] == 1:
                    newSets.append(self.dims[iset])
                    iset-= 1 ; jset -=1
            newSets.reverse()
        else:
            return KeyError("Either sets o shape have to be defined")

        return _HeaderDims(newSets)


    def transform_index(self,index_tuple):
        if not isinstance(index_tuple,tuple):
            index_tuple=(index_tuple,)

        trueLen=len([x for x in index_tuple if x is not None])
        if trueLen != self.ndim() and not Ellipsis in index_tuple:
            raise ValueError("Rank mismatch in indexing")
        if index_tuple.count(Ellipsis)>1:
            raise ValueError("Only single Ellipsis (...) allowed in indexing")

        thisIndex=[]
        for ind in index_tuple:
            if ind == Ellipsis:
                for i in range(0,self.ndim()-trueLen+1):
                    thisIndex.append(slice(None,None,None))
            elif isinstance(ind,(list,str,int,slice)) or ind is None:
                thisIndex.append(ind)
            else:
                raise TypeError("Only ...,list,str,int,slice and None allowed as indices")

        npIndex=[]
        newSets=[]

        iset=0
        for index in thisIndex:
            if index is None:
                npInd=np.newaxis
                newSet=_HeaderSet(None , 'n' , None, 1)
            else:
                setDim=self._dims[iset]
                npInd, newSet = setDim.transform_index(index)
                iset+=1
            npIndex.append(npInd)
            newSets.append(newSet)

        rankIndex=tuple([slice(None) if isinstance(ind,list) or ind is None else 0 for ind in npIndex])
        newSets = [setDim for ri, setDim in zip(rankIndex,newSets) if ri != 0]
        return self._makeNPIndex(npIndex), rankIndex, _HeaderDims(newSets)


    @staticmethod
    def _makeNPIndex(indexList):
        newinds = []
        for i, item in enumerate(indexList):
            if isinstance(item, list):
                newinds.append(item)
            elif isinstance(item,int):
                newinds.append([item])

        numpyInd = list(np.ix_(*newinds))
        newinds=[]
        for item in indexList:
            if not item is None:
                newinds.append(numpyInd.pop(0))
            else:
                newinds.append(None)

        return tuple(newinds)


