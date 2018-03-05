"""
Created on Mar 02 10:23:40 2018

.. sectionauthor:: Lyle Collins <Lyle.Collins@csiro.au>
.. codeauthor:: Lyle Collins <Lyle.Collins@csiro.au>
"""
from collections import OrderedDict
from typing import List, Union
import io
import struct
import sys

import numpy as np

import harpy.header as header
from .HeaderCommonIO import readHeader1C, readHeader7D, readHeader2D, read7DArray, read2DArray

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

V1DataTypes = ['1C', '2R', '2I', 'RE', 'RL', 'DE', 'DL']
StorageTypes = ['FULL', 'SPSE']
MaxDimVersion = [0, 7, 0, 14]


class HarFileInfo(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class HAR_IO(object):

    def __init__(self, filename):
        # type: (str) -> HAR_IO
        """
        :arg (str) fname: Name of file to open.
        :arg (char) mode: 'r': read, 'w': write or 'a': append.
        :rtype: HAR_IO
        """

        # self.endian = "="
        # self.header = "i"
        # self._HeaderDict = OrderedDict()
        self._HeaderPos = OrderedDict()
        self.readHeaderNames(filename)

    @staticmethod
    def readHeaderNames(filename: str):
        return list(HAR_IO.readHarFileInfo(filename)["headers"].keys())

    @staticmethod
    def readHarFileInfo(filename: str):
        """
        :param filename: Filename.
        :return: An `collections.OrderedDict` object, with keys as header names.
        """
        hfi = HarFileInfo()
        hfi["file"] = filename
        hfi["headers"] = OrderedDict()

        with open(filename, "rb") as f:
            f.seek(0)
            while True:
                # Read all header names
                pos, name, end_pos = HAR_IO._readHeaderPosName(f)
                if not name:
                    break
                hfi["headers"][name] = {"pos_name": pos, "pos_data": end_pos}
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
            nbyte = HAR_IO._getEntrySize(fp)
            if not nbyte: break
            data = fp.read(4).strip()
            if data:
                data += fp.read(nbyte - 4)
                HAR_IO._checkRead(fp, nbyte)
                break
            data = None
            pos = pos + 8 + nbyte
            fp.seek(pos - 4)
            HAR_IO._checkRead(fp, nbyte)
        return Hpos, fb(data), fp.tell()

    @staticmethod
    def _pass(fp: io.BufferedReader, no_bytes:int):
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
        # print("har_io._getEntrySize() tmp ", tmp)
        return tmp

    @staticmethod
    def _checkRead(fp, nbyte):
        if HAR_IO._getEntrySize(fp) != nbyte:  # Must be int at end that says how long the entry was as well...
            import traceback
            traceback.print_stack()
            raise IOError('File Corrupted, start int does not match end int ')

    @staticmethod
    def readHeader(hfi: 'HarFileInfo', header_name: str, *args, **kwargs):
        """

        :param hfi: HarFileInfo object with file information.
        :param header_name: The name of the header.
        :return: Header object with data.
        """
        header_dict = hfi["headers"][header_name]

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
             header_dict["file_dims"]) = HAR_IO._getHeaderInfo(fp, header_name)

            # # readHeader methods alter self._DataObj, self.RealDim, self.DataDimension, self.StorageType possibly self.f
            if header_dict["version"] == 1:
                header_dict["header_type"] = "data"
                if header_dict["data_type"] == '1C':
                    header_dict["array"] = HAR_IO._read1CArray(fp, file_dims=header_dict["file_dims"])
                    # if header_dict["long_name"].lower().startswith('set '):
                    #     header_dict["header_type"] = "set"
                    #     header_dict["_setNames"] = [header_dict["long_name"].split()[1]]
                elif header_dict["data_type"] == 'RE':
                    header_dict["has_elements"] = True
                    header_dict["array"] = HAR_IO._readREArray(fp, header_dict, file_dims=header_dict["file_dims"])
                # elif DataType == 'RL':
                #     readHeader7D(self, False)
                elif header_dict["data_type"] in ['2R', '2I']:
                    if header_dict["data_type"] in ['2R']:
                        data_type = 'f'
                    else:
                        data_type = 'i'

                    header_dict["array"] = HAR_IO._read2DArray(fp, data_type=data_type,
                                                               file_dims=header_dict["file_dims"],
                                                               storage_type=header_dict["storage_type"])

                else:
                    raise ValueError("Data type '%s' is unsupported." % (header_dict["data_type"]))
            else:
                raise RuntimeError("Unsupported/unrecognised HAR header version.")

        return header.HeaderArrayObj(header_dict)


    @staticmethod
    def _getHeaderInfo(fp: io.BufferedReader, name):

        secondRecordForm = "=4s2s4s70si"

        nbyte = HAR_IO._getEntrySize(fp)

        Record = HAR_IO._unpack_data(fp, secondRecordForm)

        # print("har_io.getHeaderParams() Record ", Record)

        if fb(Record[0]) != '    ':
            raise RuntimeError("Encountered characters at first four positions of 2nd Record.")

        if fb(Record[1]) in V1DataTypes:
            Version = 1
            DataType = fb(Record[1])
        else:
            Version = int(Record[1])

            if Version > 3:
                raise RuntimeError('Header "' + name + '" is HAR Version ' + fb(Record[1]) +
                                " format which cannot not be read.\nPlease check for updates of The HARpy module")

        StorageType = fb(Record[2])
        if not StorageType in StorageTypes:
            raise ValueError('Unknown StorageType "' + StorageType + '" in Header "' + name + '"')

        LongName = fb(Record[3])

        if Record[4] > MaxDimVersion[Version]:
            raise TypeError("Array Rank " + str(Record[4]) + '  in header "' + name + \
                            '" exceeds maximum permitted dimension for HAR Version' + str(Version))
        if 84 + 4 * Record[4] != nbyte:
            raise ValueError('Header "' + name + '" is corrupted at dimensions in second Record')

        # print("har_io.getHeaderParams() fp.tell() ", fp.tell())
        Sizes = HAR_IO._unpack_data(fp, "=" + ("i" * Record[-1]))

        HAR_IO._checkRead(fp, nbyte)
        return Version, DataType, StorageType, LongName, Sizes

    @staticmethod
    def _unpack_data(fp, form, data=''):
        """
        :param fp:
        :param form:
        :param data:
        :return:
        """
        # print("har_io._unpack_data() fp.tell() ", fp.tell())
        if not data:
            data = fp.read(struct.calcsize(form))
        return struct.unpack(form, data[0:struct.calcsize(form)])

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
            nbyte = HAR_IO._getEntrySize(fp)

            V = HAR_IO._unpack_data(fp, dataFormStart)

            if fb(V[0]) != '    ':
                raise IOError("Encountered characters at first four positions")

            NRec = V[1]
            if V[2] != MaxEntry:
                raise ValueError("Different Size than specified")

            if ndata + V[3] > MaxEntry:
                raise RuntimeError("More data on Header than declared")

            AllStr = fb(fp.read(V[3] * Clen))

            if nbyte != HAR_IO._getEntrySize(fp):
                raise IOError("I/O Error: sizes on 1C header do not match record length")

            for j, i in enumerate(range(0, V[3] * Clen, Clen)):
                array[ndata + j] = AllStr[i:i + Clen]
            ndata += V[3]

        return np.ascontiguousarray(array)

    @staticmethod
    def _read1CArray(fp,
                     # hd: dict,
                     file_dims=None, as_unicode:bool=as_unicode,
                     # real_dim: int = 1,
                     ):
        # hd["real_dim"] = real_dim
        array = HAR_IO._readCharVec(fp,
                                    itemsize=file_dims[1],
                                    dtype="<U12",
                                    size=(file_dims[0],),
                                    as_unicode=as_unicode)

        # array = np.chararray((file_dims[0],), itemsize=file_dims[1], unicode=as_unicode)
        # Clen = array.itemsize
        #
        # # if "<U" in str(dtype):
        # Clen = Clen // 4
        #
        # MaxEntry = file_dims[0]
        # NRec = 100
        # ndata = 0
        # dataFormStart = '=4siii'
        #
        # while NRec > 1:
        #     nbyte = HAR_IO._getEntrySize(fp)
        #
        #     V = HAR_IO._unpack_data(fp, dataFormStart)
        #
        #     if fb(V[0]) != '    ':
        #         raise IOError("Encountered characters at first four positions")
        #
        #     NRec = V[1]
        #     if V[2] != MaxEntry:
        #         raise ValueError("Different Size than specified")
        #
        #     if ndata + V[3] > MaxEntry:
        #         raise RuntimeError("More data on Header than declared")
        #
        #     AllStr = fb(fp.read(V[3] * Clen))
        #
        #     if nbyte != HAR_IO._getEntrySize(fp):
        #         raise IOError("I/O Error: sizes on 1C header do not match record length")
        #
        #     for j, i in enumerate(range(0, V[3] * Clen, Clen)):
        #         array[ndata + j] = AllStr[i:i + Clen]
        #     ndata += V[3]

        return np.ascontiguousarray(array)

    @staticmethod
    def _read2DArray(fp, data_type: str="i", storage_type: str=None, file_dims: tuple=None):
        if storage_type == 'SPSE':
            raise TypeError('Sparse storage not allowed for 2D data form.')

        if data_type in ["i", "I"]:
            data_type = np.int32
            data_type_str = "i"
        elif data_type in ["f", "F"]:
            data_type = np.float32
            data_type_str = "f"
        else:
            raise ValueError("Provided argument 'data_type' must be 'i' (integer) or 'f' (float). '%s' was provided." % data_type)

        # Note that 'order' refers to Fortran vs C (i.e. has nothing to do with floats or ints)
        array = np.ndarray(shape=file_dims[0:2], dtype=data_type, order='F')

        nrec = 50
        arraySize = array.size
        nread = 0
        while nread != arraySize:
            nbyte = HAR_IO._getEntrySize(fp)
            dataForm = "=4siiiiiii"
            V = HAR_IO._unpack_data(fp, dataForm)

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
            dat = HAR_IO._unpack_data(fp, dataForm)
            array[V[4] - 1:V[5], V[6] - 1:V[7]] = np.array(dat).reshape(xsize, ysize, order='F')

            if nbyte != HAR_IO._getEntrySize(fp):
                raise RuntimeError('Header corrupted.')

        # array = HAR_IO._read2DObj(fp, data_type, file_dims)
        array = np.ascontiguousarray(array)
        return array

    @staticmethod
    def _readREArray(fp: io.BufferedReader, header_info: dict, file_dims: tuple=None, hasSets=True):
        """

        :param fp:
        :param header_info:
        :param file_dims:
        :param hasSets:
        :return:
        """

        if hasSets:
            (header_info["coeff_name"], header_info["sets"]) = HAR_IO._readSets(fp, file_dims=file_dims)

            # print("har_io._read7DArray() set_names ", [set["name"] for set in header_info["sets"]])

            tmpDim = len(header_info["sets"])
        else:
            tmpDim = 7

        array = np.ndarray(shape=file_dims[0:tmpDim], dtype=np.float32, order='F')
        array.fill(0.0)

        # print("har_io._read7DArray() header_info[\"storage_type\"]", header_info["storage_type"])
        # print(header_info["storage_type"] == 'FULL')
        if header_info["storage_type"] == 'FULL':
            array = HAR_IO._readREFullObj(fp, array, 'f')
        else:
            array = HAR_IO._readRESparseObj(fp, array, 'f')
        return np.ascontiguousarray(array)

    @staticmethod
    def _readREFullObj(fp, array, dtype):

        nbyte = HAR_IO._getEntrySize(fp)
        dataForm = '=4sii'

        V = HAR_IO._unpack_data(fp, dataForm)

        if fb(V[0]) != '    ':
            raise Exception("Encountered characters at read7D[1]")

        nrec = V[1]
        NDim = V[2]
        dataForm = "=" + ('i' * NDim)

        V = HAR_IO._unpack_data(fp, dataForm)

        if nbyte != HAR_IO._getEntrySize(fp):
            raise RuntimeError("Header corrupted read7D[0] @ %d" % fp.tell())

        oldshape = array.shape
        array = array.flatten('F')
        idata = 0

        while nrec > 1:
            nbyte = HAR_IO._getEntrySize(fp)
            dataForm = '4s15i'
            V = HAR_IO._unpack_data(fp, dataForm)

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at first four positions at)SetEl")

            if nbyte != HAR_IO._getEntrySize(fp):
                raise RuntimeError('read7D data[2] corrupted')

            nbyte = HAR_IO._getEntrySize(fp)
            ndata = (nbyte - 8) // struct.calcsize(dtype)
            dataForm = '4si' + str(ndata) + dtype
            V = HAR_IO._unpack_data(fp, dataForm)

            if nbyte != HAR_IO._getEntrySize(fp):
                raise RuntimeError('read7D data[2])corrupted')

            if fb(V[0]) != '    ':
                raise RuntimeError("Encountered characters at read7D[2]")

            nrec = V[1]
            array[idata:idata + ndata] = V[2:]
            idata += ndata
        array = array.reshape(oldshape, order='F')

        return array

    @staticmethod
    def _readSetElementInfoRecord(fp):

        SetNames = []
        ElementList = []
        SetStatus = []

        nbyte = HAR_IO._getEntrySize(fp)

        # read the data, has to be in chunks as it is dependent on interanl size specifications
        dataForm = '=' + '4siii12si'

        # print("har_io._readSetElementInfoRecord() fp.tell() ", fp.tell())
        V = HAR_IO._unpack_data(fp, dataForm)

        if fb(V[0]) != '    ':
            raise RuntimeError("Encountered characters at first four positions at SetEl")

        NSets = V[3]
        Coefficient = fb(V[4])

        if NSets == 0:
            dataForm = '=i'
        else:
            dataForm = "=" + str(NSets * 12) + 's' + str(NSets) + 's' + str(NSets) + 'i' + 'i'

        V = HAR_IO._unpack_data(fp, dataForm)
        if NSets > 0:
            SetNames = [fb(V[0][i:i + 12]) for i in range(0, NSets * 12, 12)]
            SetStatus = [fb(V[1][i:i + 1]) for i in range(0, NSets)]

        Nexplicit = V[-1]
        dataForm = '=' + str(Nexplicit * 12) + 's'
        V = HAR_IO._unpack_data(fp, dataForm)

        if Nexplicit > 0:
            dataForm = '=' + str(Nexplicit * 12) + 's'
            V = HAR_IO._unpack_data(fp, dataForm)

            ElementList = [fb(V[-1][i:i + 12]) for i in range(0, NSets * 12, 12)]

        HAR_IO._checkRead(fp, nbyte)

        return Coefficient, SetNames, SetStatus, ElementList

    @staticmethod
    def _readSets(fp: io.BufferedReader, file_dims=None) -> dict:
        """
        :param fp: io.BufferedReader object.
        :return tuple: ("set_names", "coeff_name", "dim_desc", "dim_type")
        """
        Coefficient, SetList, SetStatus, ElementList = HAR_IO._readSetElementInfoRecord(fp)

        # print("har_io._readSets Coefficient ", Coefficient)
        # print("har_io._readSets SetList ", SetList)
        # print("har_io._readSets SetStatus ", SetStatus)
        # print("har_io._readSets ElementList ", ElementList)

        # out_dict = {"set_names": [], "dim_desc": [], "dim_type": [], "coeff_name": None}

        set_names = [name.strip() for name in SetList]
        # for name in SetList:
        #     out_dict["set_names"].append(name.strip())

        idim = 0
        header_sets = []
        processedSet = OrderedDict()
        for name, status in zip(set_names, SetStatus):
            if status == 'k':
                if name not in processedSet:
                    processedSet[name] = HAR_IO._readCharVec(fp, itemsize=12, as_unicode=as_unicode, size=tuple([file_dims[idim]]),
                                    dtype="<U12")
                header_sets.append({"name": name, "status": status, "dim_type": "Set",
                                    "dim_desc": [item.strip() for item in processedSet[name]]})
            elif status == 'u':

                header_sets.append({"name": name, "status": status, "dim_type": "Num", "dim_desc": None})
            elif status == 'e':
                header_sets.append({"name": name, "status": status, "dim_type": "El", "dim_desc": ElementList.pop(0)})
            idim += 1

        # out_dict["coeff_name"] = Coefficient

        # print("har_io._readSets set_names ", out_dict["set_names"])
        # print("har_io._readSets coeff_name ", out_dict["coeff_name"])
        # print("har_io._readSets dim_desc ", out_dict["dim_desc"])
        # print("har_io._readSets dim_type ", out_dict["dim_type"])


        return Coefficient, header_sets
