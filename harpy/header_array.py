"""
Created on Mar 02 11:39:45 2018

"""
import numpy as np
from ._header_sets import  _HeaderDims
from typing import Union,List,Dict


class HeaderArrayObj(object):

    __array_priority__ = 2 #make this precede the np __add__ operations, etc

    def __init__(self):
        self._coeff_name=""
        self._array=None
        self._sets=None
        self._long_name=""

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, obj):
        self._array = obj

    @property
    def coeff_name(self):
        return self._coeff_name

    @coeff_name.setter
    def coeff_name(self, obj):
        if not issubclass(type(obj), str):
            msg = "'obj' must be of 'str' type."
            raise TypeError(msg)
        if len(obj) < 12:
            obj.ljust(12)
        self._coeff_name = obj

    @property
    def long_name(self):
        return self._long_name

    @long_name.setter
    def long_name(self, obj):
        self._long_name = obj

    @property
    def sets(self):
        return self._sets

    @sets.setter
    def sets(self, obj):
        self._sets = obj

    @property
    def setNames(self):
        return self._sets.setNames

    @property
    def setElements(self):
        return self._sets.setElements

    @property
    def rank(self):
        return len(self.array.shape)

    def __getitem__(self, item) -> 'HeaderArrayObj':
        npInd, rankInd, newDim = self._sets.transform_index(item)
        return HeaderArrayObj.HeaderArrayFromCompiledData(array=np.array(self.array[npInd][rankInd]), SetDims=newDim)

    def __setitem__(self, key, value):
        npInd, rankInd, newDim = self._sets.transform_index(key)
        if isinstance(value,HeaderArrayObj):
            self.array[npInd] = value.array
        elif isinstance(value,(np.ndarray,int,float)):
            self.array[npInd]=value
        else:
            raise TypeError("Only HeaderArrayObj, np.ndarray, int, float  allowed in __setitem__")



    def is_valid(self, raise_exception=True) -> bool:
        """
        Checks if ``self`` is a valid ``HeaderArrayObj``.

        :param bool raise_exception: If `False`, `True`/`False` will be returned on check success/failure. Otherwise an exception is raised (the default).
        :return bool:
        """

        if not isinstance(self._array, (np.ndarray, np.float32, np.int32,np.float64)):
            if raise_exception:
                raise TypeError("HeaderArrayObj 'array' must be of type 'numpy.ndarray'.")
            else:
                return False

        if not isinstance(self._sets, _HeaderDims):
            raise TypeError("'sets' must be of type _HeaderDims")

        if not isinstance(self.long_name, str):
            raise TypeError("'long_name' must be of str type.")

        if not isinstance(self.coeff_name, str):
            raise TypeError("'coeff_name' must be of str type.")

        if self._sets.shape != self.array.shape:
            if not ( len(self._sets.shape) == 0 and self.array.size == 1):
                raise ValueError("shape of set and array do not match")
        return True

    @staticmethod
    def SetHeaderFromData(setName: str, setElements:'Union[List[str], np.array]',  long_name: str=None):
        if not isinstance(setName,str):
            raise TypeError("setName must be of type str")
        if len(setName.strip()) > 12 :
            raise ValueError("setName is restricted to 12 Characters")
        if long_name is None: long_name=""
        if isinstance(long_name, str):
            long_name="Set "+setName.strip()+" "+long_name.strip()
        else:
            raise TypeError("LongName must be string")

        if isinstance(setElements,(list,np.ndarray)):
            if not all([isinstance(el,str) for el in setElements]):
                raise TypeError("All Set Elements must be of type str")
            if not all([len(el)<=12 for el in setElements]):
                raise ValueError("Set Elelement strings must be 12 characters at most")

        if isinstance(setElements,list):
            array=np.array(setElements)
            setElDict={setName:setElements}
        elif isinstance(setElements,np.ndarray):
            array=setElements
            setElDict = {setName: setElements.tolist()}
        else:
            raise TypeError("SetElemenets must be list of str or np array of strings")


        return HeaderArrayObj.HeaderArrayFromData(array=array, long_name=long_name, sets=[setName], setElDict=setElDict)

    @staticmethod
    def HeaderArrayFromData(array: np.ndarray, coeff_name: str = None, long_name: str = None,
                            sets: 'List[str]' = None, setElDict: 'Dict[str:List[str]]' = None) -> 'HeaderArrayObj':
        """
        Creates a new HeaderArrayObj from basic data. I.e. sets and set elements are given as basic list and dict[str:list[str]]
        """

        hao = HeaderArrayObj()

        HeaderArrayObj._setHeaderBaseData(array, coeff_name, hao, long_name)
        if sets is None:
            hao.sets = _HeaderDims.fromShape(array.shape)
        else:
            if not isinstance(sets,list):
                raise TypeError("sets must be of type list")
            if not all(isinstance(setName,str) for setName in sets):
                raise TypeError("all setNames in sets must be strings")
            if not all(len(setName) <= 12 for setName in sets):
                raise TypeError("all setNames in sets must be shorter than 13 Characters")

            if setElDict is None: setElDict={}
            if not isinstance(setElDict,dict):
                raise TypeError("setElDict must be of type dict[str:list[str]]")

            hao.sets = _HeaderDims.fromSetShape(sets, setElDict, array.shape)

        if hao.is_valid():
            return hao

    @staticmethod
    def HeaderArrayFromCompiledData(array: np.ndarray, coeff_name: str=None, long_name: str=None,
                            SetDims: _HeaderDims=None) -> 'HeaderArrayObj':
        """
        Creates a new HeaderArrayObj from precompiled data. I.e. sets are already in _HeaderDim structure
        """

        hao = HeaderArrayObj()

        HeaderArrayObj._setHeaderBaseData(array, coeff_name, hao, long_name)
        if not isinstance(SetDims,_HeaderDims):
            raise TypeError("sets must be of type _HeaderDims")
        if SetDims is None:
            hao.sets = _HeaderDims.fromShape(array.shape)
        else:
            hao.sets = SetDims

        if hao.is_valid():
            return hao

    @staticmethod
    def _setHeaderBaseData(array, coeff_name, hao, long_name) -> None:
        if not isinstance(array, (np.ndarray, np.float32, np.int32,np.float64)):
            print(type(array))
            raise HeaderArrayObj.UnsupportedArrayType("'array' must be of numpy.ndarray type.")

        # Defaults handling
        if coeff_name is None:
            coeff_name = " " * 12
        if long_name is None:
            long_name = coeff_name
        if len(coeff_name) < 12:
            coeff_name = coeff_name.ljust(12)
        if len(long_name) < 70:
            long_name = long_name.ljust(70)
        hao.array = array
        hao.coeff_name = coeff_name
        hao.long_name = long_name

    def array_operation(self,
                        other: "Union[np.ndarray, HeaderArrayObj]",
                        operation: str,
                        **kwargs) -> 'HeaderArrayObj':
        """
        This method is implemented to allow for operations on the arrays of HeaderArrayObjs. Most Tablo-like
        functionality is replicated with this method.
        :param "HeaderArrayObj" other: The second ``HeaderArrayObj`` involved in the operation.
        :param str operation: A ``str`` specifying the ``numpy.ndarray`` operation attribute - e.g. ``"__add__"``.
        :param dict kwargs: Any additional kwargs are passed to the new ``HeaderArrayObj``.
        :return: A new ``HeaderArrayObj`` that results from the operation. Will have a default header name of ``"NEW1"``.
        """


        if issubclass(type(other), HeaderArrayObj):
            new_array = getattr(self.array, operation)(other.array)
            new_sets=self._sets.matchSets(sets=other._sets)
        elif issubclass(type(other), np.ndarray):
            new_array = getattr(self.array, operation)(other)
            new_sets=self._sets.matchSets(shape=other.shape)
        elif issubclass(type(other), (float, int)):
            new_array = getattr(self.array, operation)(other)
            new_sets=self._sets
        else:
            msg = "Operation is not permitted for objects that are not of 'numpy.ndarray' type, or 'HeaderArrayObj' type."
            raise TypeError(msg)

        return HeaderArrayObj.HeaderArrayFromCompiledData( array=new_array, SetDims=new_sets, **kwargs)

    def __neg__(self):
        self.array=-self.array
        return self

    def __add__(self, other):
        return self.array_operation(other, "__add__")

    def __mul__(self, other):
        return self.array_operation(other, "__mul__")

    def __truediv__(self, other):
        return self.array_operation(other, "__truediv__")

    def __floordiv__(self, other):
        return self.array_operation(other, "__floordiv__")

    def __pow__(self, other):
        return self.array_operation(other, "__pow__")

    def __mod__(self, other):
        return self.array_operation(other, "__mod__")

    def __sub__(self, other):
        return self.array_operation(other, "__sub__")

    def __radd__(self, other):
        return self.array_operation(other, "__radd__")

    def __rmul__(self, other):
        return self.array_operation(other, "__rmul__")

    def __rtruediv__(self, other):
        return self.array_operation(other, "__rtruediv__")

    def __rfloordiv__(self, other):
        return self.array_operation(other, "__rfloordiv__")

    def __rpow__(self, other):
        return self.array_operation(other, "__rpow__")

    def __rmod__(self, other):
        return self.array_operation(other, "__rmod__")

    def __rsub__(self, other):
        return self.array_operation(other, "__rsub__")

    def __str__(self) -> str:
        outputstr="\n"
        outputstr+="CoeffName".ljust(24)+": "+self.coeff_name+"\n"
        outputstr+="Rank".ljust(24)+": "+str(len(self.array.shape))+"\n"
        if self.sets:
            outputstr += self.sets.__str__()

        return outputstr

    class UnsupportedArrayType(TypeError):
        """Raised if invalid array type passed."""
        pass