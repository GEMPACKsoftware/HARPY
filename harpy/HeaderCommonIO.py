import sys

import numpy as np

__docformat__ = 'restructuredtext en'

if sys.version_info < (3,):
    as_unicode = False
else:
    as_unicode = True


def readSets(Head):
    # type: (HeaderData) -> None
    Coefficient, SetList, SetStatus, ElementList = Head.f.getSetElementInfoRecord()

    processedSet = {}

    Head._setNames = []
    Head._DimDesc = []
    Head._DimType = []

    for name in SetList:
        Head._setNames.append(name.strip())

    idim = 0
    for name, status in zip(SetList, SetStatus):
        if status == 'k':
            if not name in processedSet:
                processedSet[name] = np.chararray(tuple([Head.FileDims[idim]]), itemsize=12, unicode=as_unicode)
                Head.f.readCharVec(processedSet[name])
            Head._DimDesc.append([item.strip() for item in processedSet[name].tolist()])
            Head._DimType.append('Set')
        if status == 'u':
            Head._DimDesc.append(None)
            Head._DimType.append('Num')
        if status == 'e':
            Head._DimDesc.append(ElementList.pop(0))
            Head._DimType.append('El')
        idim += 1

    Head._Cname = Coefficient


def writeSets(Head):
    # type: (HeaderData) -> None
    Head.f.writeSetElInfo(Head._setNames, Head._DimType, Head._DimDesc, Head._Cname)


# ======================================== 1C ===============================================
def readHeader1C(Head):
    # type: (HeaderData) -> None
    Head.RealDim = 1
    Head._DataObj = np.chararray(tuple([Head.FileDims[0]]), itemsize=Head.FileDims[1], unicode=as_unicode)
    Head.f.readCharVec(Head._DataObj)

    Head.DataDimension = [Head.FileDims[0]]
    Head._DataObj = np.ascontiguousarray(Head._DataObj)


def writeHeader1C(Head):
    # type: (HeaderData) -> None
    Head.f.writeHeaderName(Head._HeaderName)
    typeString = str(Head._DataObj.dtype)
    secRecList = ['    ', '1C', 'FULL', Head._LongName, 2, Head._DataObj.size, int(typeString[2:])]
    Head.f.writeSecondRecord(secRecList)
    Head.f.write1CData(np.asfortranarray(Head._DataObj), Head._DataObj.size, int(typeString[2:]))

# ======================================== 2D ===============================================
def readHeader2D(Head, dtype):
    # type: (HeaderData,str) -> None
    if Head.StorageType == 'SPSE': raise Exception('Sparse storage not allowed on 2D data form')
    if dtype == 'f':
        Head._DataObj = np.ndarray(shape=Head.FileDims[0:2], dtype=np.float32, order='F')
    elif dtype == 'i':
        Head._DataObj = np.ndarray(shape=Head.FileDims[0:2], dtype=np.int32, order='F')

    Head.f.read2Dobject(Head._DataObj, dtype)
    Head._DataObj = np.ascontiguousarray(Head._DataObj)



def writeHeader2D(Head):
    # type: (HeaderData) -> None
    Head.f.writeHeaderName(Head._HeaderName)
    typeString = str(Head._DataObj.dtype)
    shape2D = [Head._DataObj.shape[i] if i < Head._DataObj.ndim else 1 for i in range(0, 2)]
    if typeString == 'int32':
        dtype = 'i'
        secRecList = ['    ', '2I', 'FULL', Head._LongName, 2]
    elif typeString == 'float32':
        secRecList = ['    ', '2R', 'FULL', Head._LongName, 2]
        dtype = 'f'
    secRecList.extend(shape2D)

    Head.f.writeSecondRecord(secRecList)
    Head.f.write2Dobject(np.asfortranarray(Head._DataObj), dtype)

# ======================================== 7D ===============================================
def readHeader7D(Head, hasSets=True):
    # type: (HeaderData,bool) -> None
    if hasSets:
        readSets(Head)
        tmpDim=len(Head._setNames)
    else:
        tmpDim = 7
    Head._DataObj = np.ndarray(shape=Head.FileDims[0:tmpDim], dtype=np.float32, order='F')
    Head._DataObj.fill(0.0)

    if Head.StorageType == 'FULL':
        Head._DataObj = Head.f.read7DFullObj(Head._DataObj, 'f')
    else:
        Head._DataObj = Head.f.read7DSparseObj(Head._DataObj, 'f')
    Head._DataObj=np.ascontiguousarray(Head._DataObj)


def writeHeader7D(Head):
    # type: (HeaderData) -> None
    hasElements = isinstance(Head._setNames,list) or True
    dataFill = float(np.count_nonzero(Head._DataObj)) / Head._DataObj.size

    if dataFill > 0.4:
        Head.StorageType = 'FULL'
    else:
        Head.StorageType = 'SPSE'
    shape7D = [Head._DataObj.shape[i] if i < Head._DataObj.ndim else 1 for i in range(0, 7)]

    Head.f.writeHeaderName(Head._HeaderName)
    if hasElements:
        HeaderType = 'RE'
    else:
        HeaderType = 'RL'

    secRecList = ['    ', HeaderType, Head.StorageType, Head._LongName, 7]
    secRecList.extend(shape7D)
    Head.f.writeSecondRecord(secRecList)
    if hasElements:
        Head.f.writeSetElInfo(Head._setNames, Head._DimType, Head._DimDesc, Head._Cname)

    if Head.StorageType == 'FULL':
        Head.f.write7DDataFull(np.asfortranarray(Head._DataObj), 'f')
    else:
        Head.f.write7DSparseObj(np.asfortranarray(Head._DataObj), 'f')
