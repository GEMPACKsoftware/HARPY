"""
Created on Mar 02 11:39:45 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""
import numpy as np
from harpy._header_sets import  _HeaderDims


class HeaderArrayObj(object):

    __array_priority__ = 2 #make this precede the np __add__ operations, etc

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

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
        return HeaderArrayObj.HeaderArrayFromData(array=np.array(self.array[npInd][rankInd]), sets=newDim)

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


        # if (not isinstance(self._name, str)) or (len(self._name) != 4):
        #     if raise_exception:
        #         raise HeaderArrayObj.InvalidHeaderArrayName(
        #             "Header array name (%s) must be precisely four (alphanumeric) characters." % self._name)
        #     else:
        #         return False

        if not isinstance(self._array, np.ndarray):
            if raise_exception:
                raise TypeError("HeaderArrayObj 'array' must be of type 'numpy.ndarray'.")
            else:
                return False

        if (self.sets is not None) and (not isinstance(self._sets, _HeaderDims)):
            raise TypeError("'sets' must be a list or None.")

        if not isinstance(self.long_name, str):
            raise TypeError("'long_name' must be of str type.")

        if not isinstance(self.coeff_name, str):
            raise TypeError("'coeff_name' must be of str type.")

        if not isinstance(self.version, int):
            raise TypeError("'version' must be an integer.")

        return True


    @staticmethod
    def HeaderArrayFromData(array: np.ndarray, coeff_name: str=None, long_name: str=None,
                            version: int=1, storage_type=None, file_dims=None, data_type=None,
                            sets: 'Union[None, List[dict]]'=None) -> 'HeaderArrayObj':
        """
        Creates a new HeaderArrayObj from basic data.

        :param str name: Header name (max 4 characters)
        :param numpy.ndarray array: data array.
        :param str coeff_name: coefficient name of the header array (must be no more than 12 characters).
        :param str long_name: description of content (less than or equal to 70 characters).
        :rtype: HeaderArrayObj
        """
        """
        Ignore this string - just a comment for later inclusion in method docstring
        
        
        :param list(str) sets: Name of the sets corresponding to each dimensions (size needs to be rank array)
        :param SetElements: list of list of elements (one per dim) or dict(setnames,elements)
        :type SetElements: list(list(str)) || dict(str:list(str))
        """

        hao = HeaderArrayObj()

        if not isinstance(array, np.ndarray):
            raise HeaderArrayObj.UnsupportedArrayType("'array' must be of numpy.ndarray type.")

        # Defaults handling
        if coeff_name is None:
            coeff_name = " "*12
        if long_name is None:
            long_name = coeff_name

        if len(coeff_name) < 12:
            coeff_name = coeff_name.ljust(12)
        if len(long_name) < 70:
            long_name = long_name.ljust(70)

        hao.array = array
        hao.coeff_name = coeff_name
        hao.long_name = long_name
        if sets is None:
            hao.sets = _HeaderDims.fromShape(array.shape)
        else:
            hao.sets = sets
        hao.version = version
        hao.data_type = data_type
        hao.storage_type = storage_type
        hao.file_dims = file_dims

        if hao.is_valid():
            return hao

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
        elif issubclass(type(other), (np.ndarray)):
            new_array = getattr(self.array, operation)(other)
            new_sets=self._sets.matchSets(shape=other.shape)
        elif issubclass(type(other), (float, int)):
            new_array = getattr(self.array, operation)(other)
            new_sets=self._sets
        else:
            msg = "Operation is not permitted for objects that are not of 'numpy.ndarray' type, or 'HeaderArrayObj' type."
            raise TypeError(msg)

        return HeaderArrayObj.HeaderArrayFromData( array=new_array, sets=new_sets, **kwargs)

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