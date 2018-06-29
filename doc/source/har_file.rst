HarPy API
=========

To read a header-array file (HAR) from disk, use the static method ``loadFromDisk(filename)``. This will return an instance of  ``harpy.HarFileObj()``, which is a subclass of `dict` (though all keys can be get/set as attributes).

To write a HAR file to disk, first create an ``harpy.HarFileObj()`` object, and then execute the ``writeToDisk()`` method. Those methods, and other API methods, are shown below.

.. autoclass:: harpy.HarFileObj
   :members:
