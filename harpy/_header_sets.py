"""
Created on Jun 29 14:46:48 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import numpy as np
from typing import List

class _HeaderSet():
    """
    This class is used to represent sets associated with header arrays.
    """

#    Status is unknown elements but set, element index, known set elements, no set just numeric
    _valid_status = ["u", "e", "k", "n"]
    _genSetID     = 0

    def __init__(self, name: str,
                 status: str,
                 dim_desc: str,
                 dim_size):

        self.name = name
        self.status = status
        self.dim_desc = dim_desc
        self.elemPosDict={} if self.dim_desc is None else dict(zip(range(0,len(self.dim_desc)), [elem.strip().lower() for elem in dim_desc]))
        self.dim_size = dim_size

    def transform_index(self,index):
        if isinstance(index,(str,int)):
            return self.name_to_ind(index), None

        elif isinstance(index,slice):
            newslice=self.convertSlice(index)
            SetName="XXXX" if not all(p is None for p in [newslice.start,newslice.stop,newslice.step]) else self.name
            return newslice, _HeaderSet(SetName, self.status, self.dim_desc[newslice], len(self.dim_desc[newslice]))

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
                    newslice = self.converSlice(ind)
                    npIndList.append(list(range(self.dim_size))[newslice])
                    if useElem: setElList.extend(self.dim_desc[newslice])
            if useElem:
                if len(set(setElList)) != len(setElList):
                    raise ValueError("Indexing leads to duplicate set elements which is not permitted")

            if useElem:
                return npIndList, _HeaderSet(self._newname(), self.status, setElList, len(npIndList))
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
                return self.elemPosDict[idx]
            else:
                raise ValueError("Element not in set")
        elif isinstance(idx,int):
            if idx >= self.dim_size:
                raise ValueError("Index Out Of bounds")
            return idx

    def _newname(self):
        self._genSetID+=1
        return "S@"+str(self._genSetID)


class _HeaderDims():

    def __init__(self, setList):
        self._dims=setList

    @property
    def dims(self) -> List[_HeaderSet]:
        return self._dims

    @dims.setter
    def dims(self, obj) -> None:
        self._version = obj

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

    def shape(self):
        return (sets.dim_size for sets in self._dims)

    def __str__(self):
        outputstr=""
        for set in self._dims:
            if set.status in "keu":
                outputstr+="   "+set.name.ljust(12)+": \n"
            else:
                outputstr+="   "+"Not Specified"
            if set.status in "ke":
                outputstr+="      "+", ".join(set.dim_desc)+"\n"
        return outputstr



    def compatible_shape(self,other):
        return self.shape() == other

    def transform_index(self,index_tuple):
        if not isinstance(index_tuple,tuple):
            index_tuple=(index_tuple)

        if len(index_tuple) != self.ndim() and not Ellipsis in index_tuple:
            raise ValueError("Rank mismatch in indexing")
        if index_tuple.count(Ellipsis)>1:
            raise ValueError("Only single Ellipsis (...) allowed in indexing")

        thisIndex=[]
        for ind in index_tuple:
            if ind == Ellipsis:
                for i in range(0,self.ndim()-len(index_tuple)+1):
                    thisIndex.append(slice(None,None,None))
            else:
                thisIndex.append(ind)

        npIndex=[]
        newSets=[]

        iset=0
        for index in thisIndex:
            if index is None:
                #npIndex.append(None)
                continue
            set=self._dims[iset]
            npInd, newSet = set.transform_index(index)
            npIndex.append(npInd)
            if not newSet is None: newSets.append(newSet)
            iset +=1

        return self._makeNPIndex(npIndex), _HeaderDims(newSets)


    def _makeNPIndex(self, indexList):
        if any([isinstance(item, list) for item in indexList]):
            newinds = []; shape=[]
            for i, item in enumerate(indexList):
                if isinstance(item, int):
                    newinds.append([item])
                elif isinstance(item, list):
                    newinds.append(item)
                    shape.append(len(newinds[-1]))
            numpyInd = np.ix_(*newinds)
        else:
            numpyInd = tuple(indexList)

        return numpyInd


