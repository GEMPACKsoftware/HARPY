"""
Created on Mar 02 11:39:45 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""
import numpy as np


class HeaderArrayObj(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_valid(self, raise_exception=True) -> bool:
        """
        :return:
        """

        required_keys = ["array", "name", "long_name", "data_type", "version", "storage_type", "file_dims"]
        key_present = [key in self for key in required_keys]

        if not all(key_present):
            if raise_exception:
                idx = key_present.index(False)
                raise KeyError("'%s' not in HeaderArrayObj." % required_keys[idx])
            else:
                return False

        if (not isinstance(self["name"], str)) or (len(self["name"]) != 4):
            if raise_exception:
                raise HeaderArrayObj.InvalidHeaderArrayName(
                    "Header array name (%s) must be precisely four (alphanumeric) characters." % self["name"])
            else:
                return False

        if not isinstance(self["array"], np.ndarray):
            if raise_exception:
                raise TypeError("HeaderArrayObj 'array' must be of type 'numpy.ndarray'.")
            else:
                return False

        if (self.get("sets") is not None) and (not isinstance(self["sets"], list)):
            raise TypeError("'sets' must be a list or None.")

        if not isinstance(self.get("long_name", ""), str):
            raise TypeError("'long_name' must be of str type.")

        if not isinstance(self.get("coeff_name", ""), str):
            raise TypeError("'coeff_name' must be of str type.")

        if not isinstance(self.get("version", 0), int):
            raise TypeError("'version' must be an integer.")

        return True

    def getSet(self, name: str):
        if "sets" not in self:
            raise KeyError("HeaderArrayObj does not have 'sets'.")

        for s in self["sets"]:
            if s["name"] == name:
                return s
        else:
            raise ValueError("'%s' is not the name of a set in HeaderArrayObj '%s'." % (name, self["name"]))


    @staticmethod
    def HeaderArrayFromData(name: str, array: np.ndarray, coeff_name: str=None, long_name: str=None,
                            version: int=1, storage_type=None, file_dims=None, data_type=None,
                            sets: 'Union[None, List[dict]]'=None):
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
            coeff_name = name
        if long_name is None:
            long_name = coeff_name

        # String padding if necessary
        if len(name) < 4:
            name = name.ljust(4)
        if len(coeff_name) < 12:
            coeff_name = coeff_name.ljust(12)
        if len(long_name) < 70:
            long_name = long_name.ljust(70)

        hao["name"] = name
        hao["array"] = array
        hao["coeff_name"] = coeff_name
        hao["long_name"] = long_name
        hao["sets"] = sets
        hao["version"] = version
        hao["data_type"] = data_type
        hao["storage_type"] = storage_type
        hao["file_dims"] = file_dims

        if hao.is_valid():
            return hao

    def array_operation(self, other: "HeaderArrayObj", operation: str, name: str="NEW1", **kwargs):
        """
        This method is implemented to allow for operations on the arrays of HeaderArrayObjs. Most Tablo-like
        functionality is replicated with this method.
        :param "HeaderArrayObj" other: The second ``HeaderArrayObj`` involved in the operation.
        :param str operation: A ``str`` specifying the ``numpy.ndarray`` operation attribute - e.g. ``"__add__"``.
        :param dict kwargs: Any additional kwargs are passed to the new ``HeaderArrayObj``.
        :return: A new ``HeaderArrayObj`` that results from the operation. Will have a default header name of ``"NEW1"``.
        """
        new_array = getattr(self["array"], operation)(other["array"])
        return HeaderArrayObj.HeaderArrayFromData(name="NEW1", array=new_array, **kwargs)

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

    def __matmul__(self, other):
        return self.array_operation(other, "__matmul__")

    def __sub__(self, other):
        return self.array_operation(other, "__sub__")

    class InvalidHeaderArrayName(ValueError):
        """Raised if header array name is not exactly four (alphanumeric) characters long."""
        pass

    class UnsupportedArrayType(TypeError):
        """Raised if invalid array type passed."""
        pass