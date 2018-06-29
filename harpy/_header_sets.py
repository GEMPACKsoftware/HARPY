"""
Created on Jun 29 14:46:48 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""

import numpy as np

class _HeaderSet():
    """
    This class is used to represent sets associated with header arrays.
    """

#    Status is unknown elements but set, element index, known set elements, no set just numeric
    _valid_status = ["u", "e", "k", "n"]


    def __init__(self, name: str,
                 status: str,
                 dim_desc: str,
                 dim_size):
        #super().__init__(*args, **kwargs)

        self.name = name
        self.status = status
        self.dim_desc = dim_desc
        self.dim_size = dim_size

    def transform_index(self,index):
        if isinstance(index,str):
            pass



class _HeaderDims():

    def __init__(self, setList):
        self.dims=setList

    def ndim(self):
        """
        Number of dimensions
        """
        return len(self.dims)

    def defined(self):
        """
        Tells whether dimensensions have sets defined or are just array like
        """
        return not all([dim.name is None for dim in self.dims])

    def shape(self):
        return (sets.dim_size for sets in self.dims)

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
                for i in range(0,self.ndim-len(index_tuple)+2):
                    thisIndex.append(slice(None,None,None))
            else:
                thisIndex.append(ind)

        npIndex=[]
        newSets=[]
        for set,index in zip(self.dims,index_tuple):
            npInd, newSet = set.transform_index(index)

        return tuple




    def _makeNPIndex(self, indexList):
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


