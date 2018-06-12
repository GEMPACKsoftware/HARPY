"""
Created on Mar 02 10:23:40 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""
from collections import OrderedDict
import io
import struct
import sys
import os
import math

import numpy as np

import harpy.header_array as header

# compatibility function for python 2.7/3.x
if sys.version_info < (3,):
    as_unicode = False
    def tb(x):
        return x

    def fb(x):
        return x
else:
    import codecs
    as_unicode = True

    def tb(x):
        if not x: return x
        try:
            return codecs.latin_1_encode(x)[0]
        except:
            return x

    def fb(x):
        # type: str -> str
        if not x: return x
        return x.decode('utf-8')

class HarFileInfoObj(dict):

    def __init__(self, *args, file: str=None, ha_infos: 'List[dict]'=None, **kwargs):
        """
        :param str file: The absolute path to the file.
        :param str head_arrs: A `list` of `dict`. Each `dict` object itself \
            contains the key-value pairs:
            :param "name": The name of the header-array.
            :param "pos_name": maps to an `int` that gives the byte-position of the header data.
            :param "pos_data": maps to an `int` that gives the byte-position of the array data.
        """
        super().__init__(*args, **kwargs)
        self["file"] = os.path.abspath(file)
        if ha_infos is None:
            self["ha_infos"] = []

    def addHAInfo(self, **kwargs):
        # TODO: Perform checks that there are is sufficient information before adding
        self["ha_infos"].append(kwargs)

    def is_valid(self, raise_exception=True):
        """Checks if ``self`` is a valid ``HarFileInfoObj``."""
        req_keys = ["file", "ha_infos"]
        req_keys_present = [key in self for key in req_keys]

        if not all(req_keys_present):
            if raise_exception:
                raise KeyError("Key '%s' not present in HarFileInfoObj." % req_keys[req_keys_present.index(False)])
            else:
                return False

        # TODO: Perform checks on "file" and "ha_infos" types

        return True

    def getHeaderArrayNames(self):
        return [ha_info["name"] for ha_info in self["ha_infos"]]

    def getHeaderArrayInfo(self, ha_name: str):
        idx = self._getHeaderArrayInfoIdx(ha_name)
        return self["ha_infos"][idx]

    def _getHeaderArrayInfoIdx(self, ha_name: str):

        self.is_valid()

        for idx, hai in enumerate(self["ha_infos"]):
            if hai["name"] == ha_name:
                return idx
        else:
            raise ValueError("'%s' does not exist in har file '%s'." % (ha_name, self["file"]))

class HarFileIO(object):

    V1SupDataTypes = ['1C', '2R', '2I', 'RE', 'RL', 'DE', 'DL'] # Supported data types for Version 1
    SupStorageTypes = ['FULL', 'SPSE'] # Supported storage types
    MaxDimVersion = [0, 7, 0, 14] # Maximum dimensions for each version???

    def __init__(self, filename):
        # type: (str) -> HAR_IO
        """
        :arg (str) fname: Name of file to open.
        :rtype: HAR_IO
        """

        self._HeaderPos = OrderedDict()
        self.readHeaderNames(filename)

    @staticmethod
    def readHarFileInfo(filename: str) -> 'HarFileInfoObj':
        """
        :param filename: Filename.
        :return: An `dict`, with the key-value pairs:

        """
        hfi = HarFileInfoObj(file=filename)

        with open(filename, "rb") as f:
            f.seek(0)
            while True:
                # Read all header names
                pos, name, end_pos = HarFileIO._readHeaderPosName(f)
                if not name:
                    break
                hfi.addHAInfo(**{"name": name, "pos_name": pos, "pos_data": end_pos})
                # hfi["ha_infos"].append(header.HeaderArrayObj({"name": name, "pos_name": pos, "pos_data": end_pos}))

        hfi.is_valid()
        return hfi

    @staticmethod
    def _readHeaderPosName(fp: io.BufferedReader):
        """
        Identifies position of header name and name itself.

        :param file fp: a file pointer
        :return:
        """
        # type: () -> (int, str)
        pos = fp.tell()
        data = ''
        while True:
            Hpos = pos
            nbyte = HarFileIO._getEntrySize(fp)
            if not nbyte: break
            data = fp.read(4).strip()
            if data:
                data += fp.read(nbyte - 4)
                HarFileIO._checkRead(fp, nbyte)
                break
            data = None
            pos = pos + 8 + nbyte
            fp.seek(pos - 4)
            HarFileIO._checkRead(fp, nbyte)
        return Hpos, fb(data), fp.tell()

    @staticmethod
    def readHeaderArraysFromFile(filename: str, ha_names: 'Union[None, str, List[str]]' = None):

        hfi = HarFileIO.readHarFileInfo(filename)

        if ha_names is None:
            ha_names = hfi.getHeaderArrayNames()
        elif isinstance(ha_names, str):
            ha_names = [ha_names]

        haos = []

        for ha_name in ha_names:
            haos.append(HarFileIO.readHeader(hfi=hfi, header_name=ha_name))

        return haos


    @staticmethod
    def readHeader(hfi: 'HarFileInfoObj', header_name: str, *args, **kwargs):
        """

        :param hfi: HarFileMemObj with file information.
        :param header_name: The name of the header.
        :return: Header object with data.
        """

        hfi.is_valid()
        ha_info = hfi.getHeaderArrayInfo(header_name)

        header_dict = ha_info

        with open(hfi["file"], "rb") as fp:

            try:
                fp.seek(header_dict["pos_name"])
            except KeyError:
                raise KeyError("Header '%s' does not exist in file." % header_name)

            fp.seek(header_dict["pos_data"])

            (header_dict["version"],
             header_dict["data_type"],
             header_dict["storage_type"],
             header_dict["long_name"],
             header_dict["file_dims"]) = HarFileIO._getHeaderInfo(fp, header_name)

            # # readHeader methods alter self._DataObj, self.RealDim, self.DataDimension, self.StorageType possibly self.f
            if header_dict["version"] == 1:
                header_dict["header_type"] = "data"
                if header_dict["data_type"] == '1C':
                    header_dict["array"] = HarFileIO._read1CArray(fp, file_dims=header_dict["file_dims"])
                    # if header_dict["long_name"].lower().startswith('set '):
                    #     header_dict["header_type"] = "set"
                    #     header_dict["_setNames"] = [header_dict["long_name"].split()[1]]
                elif header_dict["data_type"] == 'RE':
                    header_dict["has_elements"] = True
                    header_dict["array"] = HarFileIO._readREArray(fp, header_dict, file_dims=header_dict["file_dims"])
                # elif DataType == 'RL':
                #     readHeader7D(self, False)
                elif header_dict["data_type"] in ['2R', '2I']:
                    if header_dict["data_type"] in ['2R']:
                        data_type = 'f'
                    else:
                        data_type = 'i'

                    header_dict["array"] = HarFileIO._read2DArray(fp, data_type=data_type,
                                                               file_dims=header_dict["file_dims"],
                                                               storage_type=header_dict["storage_type"])

                else:
                    raise ValueError("Data type '%s' is unsupported." % (header_dict["data_type"]))
            else:
                raise RuntimeError("Unsupported/unrecognised HAR header version.")

        return header.HeaderArrayObj(header_dict)

    @staticmethod
    def _read1CArray(fp, file_dims=None, as_unicode: bool = as_unicode, ):
        array = HarFileIO._readCharVec(fp,
                                       itemsize=file_dims[1],
                                       dtype="<U12",
                                       size=(file_dims[0],),
                                       as_unicode=as_unicode)

        return np.ascontiguousarray(array)

    @staticmethod
    def _read2DArray(fp, data_type: str = "i", storage_type: str = None, file_dims: tuple = None):
        if storage_type == 'SPSE':
            raise TypeError('Sparse storage not allowed for 2D data form.')

        if data_type in ["i", "I"]:
            data_type = np.int32
            data_type_str = "i"
        elif data_type in ["f", "F"]:
            data_type = np.float32
            data_type_str = "f"
        else:
            raise ValueError(
                "Provided argument 'data_type' must be 'i' (integer) or 'f' (float). '%s' was provided." % data_type)

        # Note that 'order' refers to Fortran vs C (i.e. has nothing to do with floats or ints)
        array = np.ndarray(shape=file_dims[0:2], dtype=data_type, order='F')

        nrec = 50
        arraySize = array.size
        nread = 0
        while nread != arraySize:
            nbyte = HarFileIO._getEntrySize(fp)
            dataForm = "=4siiiiiii"
            V = HarFileIO._unpack_data(fp, dataForm)

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at read2D loop")
            if V[2] != array.shape[0]:
                raise ValueError("Mismatching row sizes on header")
            if V[3] != array.shape[1]:
                raise ValueError("Mismatching col sizes on header")

            xsize = V[5] - V[4] + 1
            ysize = V[7] - V[6] + 1
            ndata = xsize * ysize
            nread += ndata
            dataForm = "=" + str(ndata) + data_type_str
            dat = HarFileIO._unpack_data(fp, dataForm)
            array[V[4] - 1:V[5], V[6] - 1:V[7]] = np.array(dat).reshape(xsize, ysize, order='F')

            if nbyte != HarFileIO._getEntrySize(fp):
                raise RuntimeError('Header corrupted.')

        # array = HAR_IO._read2DObj(fp, data_type, file_dims)
        array = np.ascontiguousarray(array)
        return array

    @staticmethod
    def _readREArray(fp: io.BufferedReader, header_info: dict, file_dims: tuple = None, hasSets=True):
        """

        :param fp:
        :param header_info:
        :param file_dims:
        :param hasSets:
        :return:
        """

        if hasSets:
            (header_info["coeff_name"], header_info["sets"]) = HarFileIO._readSets(fp, file_dims=file_dims)

            # print("har_io._read7DArray() set_names ", [set["name"] for set in header_info["sets"]])

            tmpDim = len(header_info["sets"])
        else:
            tmpDim = 7

        array = np.ndarray(shape=file_dims[0:tmpDim], dtype=np.float32, order='F')
        array.fill(0.0)

        # print("har_io._read7DArray() header_info[\"storage_type\"]", header_info["storage_type"])
        # print(header_info["storage_type"] == 'FULL')
        if header_info["storage_type"] == 'FULL':
            array = HarFileIO._readREFullObj(fp, array, 'f')
        else:
            array = HarFileIO._readRESparseObj(fp, array, 'f')
        return np.ascontiguousarray(array)


    @staticmethod
    def _getHeaderInfo(fp: io.BufferedReader, name):

        nbyte = HarFileIO._getEntrySize(fp)

        secondRecordForm = "=4s2s4s70si"
        Record = HarFileIO._unpack_data(fp, secondRecordForm)

        if fb(Record[0]) != '    ':
            raise RuntimeError("Encountered characters at first four positions of 2nd Record.")

        if fb(Record[1]) in HarFileIO.V1SupDataTypes:
            Version = 1
            DataType = fb(Record[1])
        else:
            Version = int(Record[1])

            if Version > 3:
                raise RuntimeError('Header "' + name + '" is HAR Version ' + fb(Record[1]) +
                                " format which cannot not be read.\nPlease check for updates of The HARpy module")

        StorageType = fb(Record[2])
        if not StorageType in HarFileIO.SupStorageTypes:
            raise ValueError('Unknown StorageType "' + StorageType + '" in Header "' + name + '"')

        LongName = fb(Record[3])

        if Record[4] > HarFileIO.MaxDimVersion[Version]:
            raise TypeError("Array Rank " + str(Record[4]) + '  in header "' + name + \
                            '" exceeds maximum permitted dimension for HAR Version' + str(Version))
        if 84 + 4 * Record[4] != nbyte:
            raise ValueError('Header "' + name + '" is corrupted at dimensions in second Record')

        Sizes = HarFileIO._unpack_data(fp, "=" + ("i" * Record[-1]))

        HarFileIO._checkRead(fp, nbyte)
        return Version, DataType, StorageType, LongName, Sizes

    @staticmethod
    def _readCharVec(fp: io.BufferedReader, itemsize:int=None, dtype=None, size:tuple=None, as_unicode=as_unicode):
        """

        :param fp:
        :param itemsize:
        :param dtype:
        :param size:
        :param as_unicode:
        :return:
        """

        array = np.chararray(size, itemsize=itemsize, unicode=as_unicode)
        Clen = array.itemsize

        if "<U" in str(dtype):
            Clen = Clen // 4

        MaxEntry = size[0]
        NRec = 100
        ndata = 0
        dataFormStart = '=4siii'

        while NRec > 1:
            nbyte = HarFileIO._getEntrySize(fp)

            V = HarFileIO._unpack_data(fp, dataFormStart)

            if fb(V[0]) != '    ':
                raise IOError("Encountered characters at first four positions")

            NRec = V[1]
            if V[2] != MaxEntry:
                raise ValueError("Different Size than specified")

            if ndata + V[3] > MaxEntry:
                raise RuntimeError("More data on Header than declared")

            AllStr = fb(fp.read(V[3] * Clen))

            if nbyte != HarFileIO._getEntrySize(fp):
                raise IOError("I/O Error: sizes on 1C header do not match record length")

            for j, i in enumerate(range(0, V[3] * Clen, Clen)):
                array[ndata + j] = AllStr[i:i + Clen]
            ndata += V[3]

        return np.ascontiguousarray(array)

    @staticmethod
    def _readREFullObj(fp, array, dtype):

        nbyte = HarFileIO._getEntrySize(fp)
        dataForm = '=4sii'

        V = HarFileIO._unpack_data(fp, dataForm)

        if fb(V[0]) != '    ':
            raise Exception("Encountered characters at read7D[1]")

        nrec = V[1]
        NDim = V[2]
        dataForm = "=" + ('i' * NDim)

        V = HarFileIO._unpack_data(fp, dataForm)

        if nbyte != HarFileIO._getEntrySize(fp):
            raise RuntimeError("Header corrupted read7D[0] @ %d" % fp.tell())

        oldshape = array.shape
        array = array.flatten('F')
        idata = 0

        while nrec > 1:
            nbyte = HarFileIO._getEntrySize(fp)
            dataForm = '4s15i'
            V = HarFileIO._unpack_data(fp, dataForm)

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at first four positions at)SetEl")

            if nbyte != HarFileIO._getEntrySize(fp):
                raise RuntimeError('read7D data[2] corrupted')

            nbyte = HarFileIO._getEntrySize(fp)
            ndata = (nbyte - 8) // struct.calcsize(dtype)
            dataForm = '4si' + str(ndata) + dtype
            V = HarFileIO._unpack_data(fp, dataForm)

            if nbyte != HarFileIO._getEntrySize(fp):
                raise RuntimeError('read7D data[2])corrupted')

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at read7D[2]")

            nrec = V[1]
            array[idata:idata + ndata] = V[2:]
            idata += ndata
        array = array.reshape(oldshape, order='F')

        return array

    @staticmethod
    def _readRESparseObj(fp:io.BufferedReader, array: np.ndarray, dtype):
        nbyte = HarFileIO._getEntrySize(fp)
        dataForm = '=4siii80s'
        nrec = 50
        V = HarFileIO._unpack_data(fp, dataForm)
        NNonZero = V[1]
        if V[2] != 4:
            raise ValueError("Can only read integer 4 in read7DSparse7D ")
        if V[3] != 4:
            raise ValueError("Can only read real 4 in read7DSparse7D ")
        if nbyte != HarFileIO._getEntrySize(fp):
            raise ValueError('Header corrupted read7DSparse[0]')

        oldshape = array.shape

        array = array.flatten('F')

        while nrec > 1:
            nbyte = HarFileIO._getEntrySize(fp)
            dataForm = '=4siii'
            V = HarFileIO._unpack_data(fp, dataForm)

            if fb(V[0]) != '    ':
                raise ValueError("Encountered characters at read7DSparse loop")

            nrec = V[1]
            NHere = V[3]
            dataForm = '=' + str(NHere) + 'i' + str(NHere) + dtype
            V = HarFileIO._unpack_data(fp, dataForm)

            if nbyte != HarFileIO._getEntrySize(fp):
                raise ValueError('Header corrupted read7DSparse)[1]')

            for i in range(0, NHere):
                array[V[i] - 1] = V[i + NHere]

        array = array.reshape(oldshape, order='F')
        return array

    @staticmethod
    def _readSetElementInfoRecord(fp):

        SetNames = []
        ElementList = []
        SetStatus = []

        nbyte = HarFileIO._getEntrySize(fp)

        # read the data, has to be in chunks as it is dependent on interanl size specifications
        dataForm = '=' + '4siii12si'

        V = HarFileIO._unpack_data(fp, dataForm)

        if fb(V[0]) != '    ':
            raise RuntimeError("Encountered characters at first four positions at SetEl")

        NSets = V[3]
        Coefficient = fb(V[4])

        if NSets == 0:
            dataForm = '=i'
        else:
            dataForm = "=" + str(NSets * 12) + 's' + str(NSets) + 's' + str(NSets) + 'i' + 'i'

        V = HarFileIO._unpack_data(fp, dataForm)
        if NSets > 0:
            SetNames = [fb(V[0][i:i + 12]) for i in range(0, NSets * 12, 12)]
            SetStatus = [fb(V[1][i:i + 1]) for i in range(0, NSets)]

        Nexplicit = V[-1]
        dataForm = '=' + str(Nexplicit * 12) + 's'
        V = HarFileIO._unpack_data(fp, dataForm)

        if Nexplicit > 0:
            dataForm = '=' + str(Nexplicit * 12) + 's'
            V = HarFileIO._unpack_data(fp, dataForm)

            ElementList = [fb(V[-1][i:i + 12]) for i in range(0, NSets * 12, 12)]

        HarFileIO._checkRead(fp, nbyte)

        return Coefficient, SetNames, SetStatus, ElementList

    @staticmethod
    def _readSets(fp: io.BufferedReader, file_dims=None) -> dict:
        """
        :param fp: io.BufferedReader object.
        :return tuple: (coefficient_name, header_sets)
        """
        Coefficient, SetList, SetStatus, ElementList = HarFileIO._readSetElementInfoRecord(fp)

        set_names = [name.strip() for name in SetList]

        idim = 0
        header_sets = []
        processedSet = OrderedDict()
        for name, status in zip(set_names, SetStatus):
            if status == 'k':
                if name not in processedSet:
                    processedSet[name] = HarFileIO._readCharVec(fp, itemsize=12, as_unicode=as_unicode, size=tuple([file_dims[idim]]),
                                    dtype="<U12")
                header_sets.append({"name": name, "status": status, "dim_type": "Set",
                                    "dim_desc": [item.strip() for item in processedSet[name]]})
            elif status == 'u':
                header_sets.append({"name": name, "status": status, "dim_type": "Num", "dim_desc": None})
            elif status == 'e':
                header_sets.append({"name": name, "status": status, "dim_type": "El", "dim_desc": ElementList.pop(0)})
            idim += 1

        return Coefficient, header_sets

    @staticmethod
    def _unpack_data(fp, form, data=''):
        """
        :param fp:
        :param form:
        :param data:
        :return:
        """
        # print("HarFileIO._unpack_data() form, data ", form, data)
        if not data:
            data = fp.read(struct.calcsize(form))
        return struct.unpack(form, data[0:struct.calcsize(form)])

    @staticmethod
    def _pass(fp: io.BufferedReader, no_bytes: int):
        data = fp.read(no_bytes)
        return

    @staticmethod
    def _getEntrySize(fp: io.BufferedReader) -> int:
        """
        Reads 4 bytes (corresponds to the size of the entry).
        :param fp: Read directly from ``fp``.
        """
        data = fp.read(4)
        if not data:
            return None
        tmp = struct.unpack("=i", data)[0]
        return tmp

    @staticmethod
    def _checkRead(fp, nbyte):
        if HarFileIO._getEntrySize(fp) != nbyte:  # Must be int at end that says how long the entry was as well...
            import traceback
            traceback.print_stack()
            raise IOError('File Corrupted, start int does not match end int.')

    @staticmethod
    def writeHeaders(filename: 'Union[str, io.BufferedWriter]',
                     head_arr_objs: 'Union[header.HeaderArrayObj, List[header.HeaderArrayObj]]'):
        """
        :param filename: name of file to write into.
        # :param fp: file object to write into.
        :param header_name: Name of header. Must be precisely four characters.
        :param header_longname: Long name. Must be precisely 70 characters.
        :param header_array: A `numpy.ndarray` object of the data.
        :return:
        """
        if isinstance(head_arr_objs, header.HeaderArrayObj):
            head_arr_objs = [head_arr_objs]

        for head_arr_obj in head_arr_objs:
            if not isinstance(head_arr_obj, header.HeaderArrayObj):
                raise TypeError("All 'head_arr_objs' must be of header.HeaderArrayObj type.")
            head_arr_obj.is_valid()

        if isinstance(filename, str):
            fp = open(filename, "wb")
        # elif issubclass(type(filename), io.BufferedWriter):
        #     fp = filename
        else:
            raise TypeError("'filename' is invalid - must be either file object or string.")

        with fp:
            for head_arr_obj in head_arr_objs:
                header_type_str = str(head_arr_obj["array"].dtype)
                has_sets = "sets" in head_arr_obj
                # HarFileIO._writeHeader(fp, head_arr_obj)
                if 'float32' == header_type_str and (head_arr_obj["array"].ndim != 2 or has_sets):
                    HarFileIO._writeHeader7D(fp, head_arr_obj)
                elif 'int32' == header_type_str or 'float32' == header_type_str:
                    HarFileIO._writeHeader2D(fp, head_arr_obj)
                elif '<U' in header_type_str or '|S' in header_type_str:
                    if head_arr_obj["array"].ndim > 1:
                        print('"' + head_arr_obj["name"] + '" can not be written as character arrays ndim>1 are not yet supported')
                        return
                    HarFileIO._writeHeader1C(fp, head_arr_obj)
                else:
                    raise TypeError('Can not write data in Header: "' +
                                    head_arr_obj["name"] + '" as data style does not match any known Header type')
                fp.flush()

    @staticmethod
    def _writeHeader(fp: io.BufferedReader, head_arr_obj: header.HeaderArrayObj):

        head_arr_obj["storage_type"] = 'FULL'

        header_type_str = str(head_arr_obj["array"].dtype)
        if header_type_str == "float32":
            type_char = 'f'
            if head_arr_obj["array"].ndim > 2:
                if (float(np.count_nonzero(head_arr_obj["array"])) / head_arr_obj["array"].size) <= 0.4:
                    head_arr_obj["storage_type"] = 'SPSE'
                max_dim = 7
                if "sets" in head_arr_obj:
                    head_arr_obj["data_type"] = "RE"
                else:
                    head_arr_obj["data_type"] = "RL"
            elif head_arr_obj["array"].ndim == 2:
                max_dim = 2
                head_arr_obj["data_type"] = "2R"
        elif header_type_str == "int32":
            type_char = 'i'
            max_dim = 2
            head_arr_obj["data_type"] = "2I"
        elif '<U' in header_type_str or '|S' in header_type_str:
            if head_arr_obj["array"].ndim > 1:
                raise ValueError("'%s' can not be written as character arrays with more than 1 dimension are not yet supported." % head_arr_obj["name"])
            max_dim = 2 # Yes, seems a bit counter-intuitive I know
            head_arr_obj["data_type"] = "1C"
        else:
            raise TypeError("Can not write data in '%s' as array does not match any known type." % head_arr_obj["name"])

        secRecList = ['    ', head_arr_obj["data_type"], head_arr_obj["storage_type"], head_arr_obj["long_name"], max_dim]
        ext = [head_arr_obj["array"].shape[i] if i < head_arr_obj["array"].ndim else 1 for i in range(max_dim)]
        if head_arr_obj["data_type"] == "1C":
            ext = [head_arr_obj["array"].size, int(header_type_str[2:])]
        secRecList = secRecList + ext

        HarFileIO._writeHeaderName(fp, head_arr_obj["name"])
        HarFileIO._writeSecondRecord(fp, secRecList)

        if head_arr_obj["data_type"] in ["RE", "RL"]:
            if head_arr_obj["storage_type"] == 'FULL':
                HarFileIO._write7DFullArray(fp, np.asfortranarray(head_arr_obj["array"]), type_char)
            else:
                HarFileIO._write7DSparseArray(fp, np.asfortranarray(head_arr_obj["array"]), type_char)
        elif head_arr_obj["data_type"] in ["2I", "2R"]:
            HarFileIO._write2DArray(fp, np.asfortranarray(head_arr_obj["array"]), type_char)
        elif head_arr_obj["data_type"] in ["1C"]:
            HarFileIO._write1CArray(fp, np.asfortranarray(head_arr_obj["array"]), head_arr_obj["array"].size, int(header_type_str[2:]))
        else:
            raise ValueError("Unknown 'data_type' for this HeaderArrayObj.")

    @staticmethod
    def _writeHeader7D(fp: io.BufferedReader, head_arr_obj: header.HeaderArrayObj):
        hasElements = isinstance(head_arr_obj["sets"], list)
        dataFill = float(np.count_nonzero(head_arr_obj["array"])) / head_arr_obj["array"].size

        if dataFill > 0.4:
            head_arr_obj["storage_type"] = 'FULL'
        else:
            head_arr_obj["storage_type"] = 'SPSE'

        shape7D = [head_arr_obj["array"].shape[i] if i < head_arr_obj["array"].ndim else 1 for i in range(0, 7)]

        HarFileIO._writeHeaderName(fp, head_arr_obj["name"])
        if hasElements:
            HeaderType = 'RE'
        else:
            HeaderType = 'RL'

        secRecList = ['    ', HeaderType, head_arr_obj["storage_type"], head_arr_obj["long_name"], 7]
        secRecList.extend(shape7D)
        HarFileIO._writeSecondRecord(fp, secRecList)
        if hasElements:
            HarFileIO._writeSetElInfo(fp, head_arr_obj)

        if head_arr_obj["storage_type"] == 'FULL':
            HarFileIO._write7DFullArray(fp, np.asfortranarray(head_arr_obj["array"]), 'f')
        else:
            HarFileIO._write7DSparseArray(fp, np.asfortranarray(head_arr_obj["array"]), 'f')

    @staticmethod
    def _writeHeader2D(fp: io.BufferedReader, head_arr_obj: header.HeaderArrayObj):
        HarFileIO._writeHeaderName(fp, head_arr_obj["name"])
        typeString = str(head_arr_obj["array"].dtype)
        shape2D = [head_arr_obj["array"].shape[i] if i < head_arr_obj["array"].ndim else 1 for i in range(0, 2)]
        if typeString == 'int32':
            dtype = 'i'
            secRecList = ['    ', '2I', 'FULL', head_arr_obj["long_name"], 2]
        elif typeString == 'float32':
            secRecList = ['    ', '2R', 'FULL', head_arr_obj["long_name"], 2]
            dtype = 'f'
        secRecList.extend(shape2D)

        HarFileIO._writeSecondRecord(fp, secRecList)
        HarFileIO._write2DArray(fp, np.asfortranarray(head_arr_obj["array"]), dtype)

    @staticmethod
    def _writeHeader1C(fp: io.BufferedReader, head_arr_obj: header.HeaderArrayObj):

        HarFileIO._writeHeaderName(fp, head_arr_obj["name"])
        typeString = str(head_arr_obj["array"].dtype)
        no_chars = int(typeString[2:])
        secRecList = ['    ', '1C', 'FULL', head_arr_obj["long_name"], 2, head_arr_obj["array"].size, no_chars]
        HarFileIO._writeSecondRecord(fp, secRecList)
        HarFileIO._write1CArray(fp, np.asfortranarray(head_arr_obj["array"]), head_arr_obj["array"].size, no_chars)

    @staticmethod
    def _writeHeaderName(fp: io.BufferedReader, name: str):

        if len(name) != 4:
            raise ValueError('Header name ' + name + ' is not 4 characters long. Header array not written to file.')

        dataForm = '=i4si'
        packed = struct.pack(dataForm, 4, tb(name), 4)
        fp.write(packed)

    @staticmethod
    def _writeSecondRecord(fp: io.BufferedReader, List):
        nint = len(List) - 4

        if len(List[3]) != 70:
            raise ValueError("'long_name' must be precisely 70 characters long. 'long_name' is: %s (%d characters long)." % (List[3], len(List[3])))

        List = [tb(x) if isinstance(x, str) else x for x in List]
        dataForm = '=i4s2s4s70s' + 'i' * nint + 'i' # For reading it is "=4s2s4s70si"
        # print("HarFileIO._writeSecondRecord() dataForm ", dataForm)
        byteLen = struct.calcsize(dataForm) - 8
        List.append(byteLen)
        List.insert(0, byteLen)
        packed = struct.pack(dataForm, *List)
        # print("HarFileIO._writeSecondRecord() ", len(packed))
        fp.write(packed)

    @staticmethod
    def _write7DFullArray(fp, array, dtype):
        StEndList = []
        for i, j in HarFileIO._slice_inds(array, 7996):
            StEndList.append([i[:], j[:]])
        nrec = len(StEndList) * 2 + 1
        dataForm = '=i4sii7ii'
        nbyte = struct.calcsize(dataForm) - 8
        writeList = [nbyte, tb('    '), nrec, 7]
        writeList.extend([array.shape[i] if i < array.ndim else 1 for i in range(0, 7)])
        writeList.append(nbyte)

        fp.write(struct.pack(dataForm, *writeList))

        array1=array.flatten('F')
        nWritten=0
        for StEnd in StEndList:
            nrec = nrec - 1
            st = StEnd[0];
            end = StEnd[1]

            PosList = [[st[i] + 1, end[i]][ind] if i < array.ndim else [1, 1][ind] for i in range(0, 7) for ind in
                       range(0, 2)]
            dataForm = '=i4s16i'
            nbyte = struct.calcsize(dataForm) - 8
            writeList = [nbyte, tb('    '), nrec]
            writeList.extend(PosList)
            writeList.append(nbyte)
            fp.write(struct.pack(dataForm, *writeList))

            nrec = nrec - 1
            ndata = 1
            for i, j in zip(st, end): ndata *= (j - i)
            if dtype == 'f' or dtype == 'i': nbyte = ndata * 4 + 8

            dataForm = '=i4si'
            fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec))
            dataForm = '=' + str(ndata) + dtype
            fp.write(struct.pack(dataForm, *array1[nWritten:nWritten+ndata].flatten('F')))
            nWritten+=ndata
            dataForm = '=i'
            fp.write(struct.pack(dataForm, nbyte))

    @staticmethod
    def _write7DSparseArray(fp, array, dtype):
        NNonZero = np.count_nonzero(array)
        Comment = 80 * ' '
        dataForm = '=i4siii80si'
        fp.write(struct.pack(dataForm, 96, tb('    '), NNonZero, 4, 4, tb(Comment), 96))
        maxData = 3996
        nrec = (NNonZero - 1) // maxData + 1
        ndata = 0
        valList = []
        indexList = []

        if NNonZero == 0:
            fp.write(struct.pack('=i4siiii', 16, tb('    '), 1, 0, 0, 16))
            return

        indexList=maxData*[None]
        valList=maxData*[None]
        tmp=array.flatten('F')
        nzind=np.nonzero(tmp)
        for i in nzind[0]:
            ndata += 1
            indexList[ndata-1]=i+1
            valList[ndata-1]=tmp[i]
            if ndata == maxData:
                HarFileIO._writeSparseList(fp, NNonZero, dtype, indexList, ndata, nrec, valList)
                nrec = nrec - 1
                ndata = 0

        if ndata != 0:
            indexList=indexList[0:ndata]
            valList=valList[0:ndata]
            HarFileIO._writeSparseList(fp, NNonZero, dtype, indexList, ndata, nrec, valList)

    @staticmethod
    def _write2DArray(fp, array, dtype):
        dataForm = "=i4siiiiiii"
        maxData = 7991
        nrec = (array.size - 1) // maxData + 1
        ndata = 0
        for st, end in HarFileIO._slice_inds(array, maxData):
            if array.ndim == 1:
                indexTuple = (slice(st[0], end[0]))
                ndata = (end[0] - st[0])
                nbyte = ndata * 4 + 32
                fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec, array.size, 1, st[0] + 1, end[0], 1, 1))
            elif array.ndim == 2:
                indexTuple = (slice(st[0], end[0]), slice(st[1], end[1]))
                ndata = (end[0] - st[0]) * (end[1] - st[1])
                nbyte = ndata * 4 + 32
                fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec, array.shape[0],
                                         array.shape[1], st[0] + 1, end[0], st[1] + 1, end[1]))

            dataForm1 = '=' + str(ndata) + dtype
            fp.write(struct.pack(dataForm1, *array[indexTuple].flatten('F')))
            fp.write(struct.pack('=i', nbyte))
            nrec = nrec - 1

    @staticmethod
    def _write1CArray(fp, array, vecDim, strLen):
        maxwrt = 29996
        maxPerLine = maxwrt // strLen
        nrec = (vecDim - 1) // maxPerLine + 1
        nLeft = vecDim
        nDone = 0
        if nrec==0:
            dataForm = '=i4siiii'
            fp.write(struct.pack(dataForm, 16, tb('    '), 1, 0, 0,16))
        while nrec > 0:
            dataForm = '=i4siii'
            nOnRec = min(nLeft, maxPerLine)
            ndata = 16 + nOnRec * strLen
            fp.write(struct.pack(dataForm, ndata, tb('    '), nrec, vecDim, nOnRec))
            dataForm = '=' + str(ndata - 16) + 'si'
            packStr = tb(''.join([array[i].ljust(strLen) for i in range(nDone,nDone + nOnRec)]))
            fp.write(struct.pack(dataForm, packStr, ndata))
            nrec -= 1
            nLeft -= nOnRec
            nDone += nOnRec

    @staticmethod
    def _writeSparseList(fp, NNonZero, dtype, indexList, ndata, nrec, valList):
        dataForm = '=i4siii'
        nbyte = ndata * 2 * 4 + 16
        fp.write(struct.pack(dataForm, nbyte, tb('    '), nrec, NNonZero, ndata))
        dataForm = '=' + str(ndata) + 'i'
        fp.write(struct.pack(dataForm, *indexList))
        dataForm = '=' + str(ndata) + dtype
        fp.write(struct.pack(dataForm, *valList))
        fp.write(struct.pack('=i', nbyte))

    @staticmethod
    def _writeSetElInfo(fp,
                        # sets,
                        # indexTypes,
                        # Elements,
                        # CName,
                        header_arr_obj: header.HeaderArrayObj):
        sets = [set["name"] for set in header_arr_obj["sets"]]
        indexTypes = [set["dim_type"] for set in header_arr_obj["sets"]]
        Elements = [set["dim_desc"] for set in header_arr_obj["sets"]]
        CName = header_arr_obj["coeff_name"]
        tmp = {}
        elList = []
        if not sets:
            nToWrite = 0
            nSets = 0
            nElement = 0
        else:
            statusStr = ''
            outputElements = []
            for i, j, setEls in zip(sets, indexTypes, Elements):
                if j == 'Set':
                    if not i in tmp:
                        outputElements.append(setEls)
                    tmp[i] = setEls
                    statusStr += 'k'
                elif j == 'El':
                    elList.append(setEls[0])
                    statusStr += 'e'
                else:
                    statusStr += 'u'
            nToWrite = len(tmp)
            nElement = len(elList)
            ElementStr = tb(''.join(elList))
            statusStr = tb(statusStr)
            SetStr = tb(''.join([item.ljust(12) for item in sets]))
            nSets = len(sets)

        dataForm = '=i4siii12si'
        if nSets > 0: dataForm += str(nSets * 13) + 's' + str(nSets) + 'i'
        dataForm += 'i'
        if nElement > 0: dataForm += str(nElement * 12)
        dataForm += 'i'
        nbyte = struct.calcsize(dataForm) - 8
        writeList = [nbyte, tb('    '), nToWrite, 1, nSets, tb(CName.ljust(12)), 1]
        if nSets > 0:
            writeList.append(SetStr + statusStr)
            writeList.extend([0 for i in Elements])

        writeList.append(nElement)
        if nElement > 0: writeList.append(ElementStr)
        writeList.append(nbyte)

        fp.write(struct.pack(dataForm, *writeList))

        if nToWrite > 0:
            for Els in outputElements:
                array = np.array(Els)
                HarFileIO._write1CArray(fp, array, len(Els), 12)

    @staticmethod
    def _slice_inds(a, size):
        if a.ndim==0:
            yield [0],[1]
            return
        stride = [i // a.dtype.itemsize for i in a.strides]
        offset = [i * j for i, j in zip(stride, a.shape)]
        ndim = len(offset)
        for inc_dim, off in enumerate(offset):
            if off > size: break
        nslice = size // stride[inc_dim]
        increment = int(size // stride[inc_dim])
        tot_iter = int(math.ceil(float(a.shape[inc_dim]) / nslice) * offset[-1] / offset[inc_dim])

        end_index = [0 if i == inc_dim else 1 for i in range(0, ndim)]
        end_index[0:inc_dim] = a.shape[0:inc_dim]

        start_index = [0 for i in range(0, ndim)]

        for i in range(0, tot_iter):
            if end_index[inc_dim] == a.shape[inc_dim]:
                start_index[inc_dim] = 0
                end_index[inc_dim] = increment
                for j in range(inc_dim + 1, ndim):
                    if end_index[j] == a.shape[j]:
                        start_index[j] = 0
                        end_index[j] = 1
                    else:
                        start_index[j] = end_index[j]
                        end_index[j] += 1
                        break
            else:
                start_index[inc_dim] = end_index[inc_dim]
                end_index[inc_dim] = min(a.shape[inc_dim], end_index[inc_dim] + increment)
            yield start_index, end_index