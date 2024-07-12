from __future__ import print_function, absolute_import
from .har_file import HarFileObj
from .header_array import HeaderArrayObj
import numpy as np


class SL4(object):

    def __init__(self, fname, extractList=None):
        """
        Decodes an SL4 file and returns variable Header (1 dim add containing cumulative solution + subtotals if present)
        In addtion Set information is available via the getSet functions
        :param fname: str filename of the SL4 file
        :param extractList: list(varNames) only required if a subset of variables should be extracted
        """
        self._sets=[]
        self._variables=[]
        self.setHeaders={}
        self.variableDict={}
        self.varTypeDict={}
        self.solFile=HarFileObj("fname"+".sol")
        self.decode_SL4(fname, extractList)


    @property
    def variableNames(self):
        return self._variables
    @variableNames.setter
    def variableNames(self, val):
        raise Exception ("Variables in SL4 class cannot be assigned")

    @property
    def setNames(self):
        return self._sets
    @setNames.setter
    def setNames(self,val):
        raise Exception ("Sets in SL4 class cannot be assigned")

    def varType(self,name):
        if not name.strip().lower() in self.varTypeDict:
            raise Exception("Could not find variable '" + name + "' in the SL4 file. Please check the variable list")
        return self.varTypeDict[name.strip().lower()]

    def getSet(self,name):
        if not name.strip().lower() in self.setHeaders:
            raise Exception ("Could not find set '"+name+"' in the SL4 file. Please check the set list")
        return self.setHeaders[name.strip().lower()]

    def getVariable(self,name:str) -> HeaderArrayObj:
        if not name.strip().lower() in self.variableDict:
            raise Exception("Could not find variable '" + name + "' in the SL4 file. Please check the variable list")
        return self.variableDict[name.strip().lower()]

    def decode_SL4(self, fname, extractList):
        HarObj=HarFileObj(fname)

        #collect the set information. Not sure whether intertemporals work yet
        setNames =   HarObj["STNM"]
        self.generateSetHeaders(HarObj, setNames)
        resultsSet=["Cumulative"]
        resultsDataHeaders=["CUMS"]
        resultsShockComponents = ["SHCK"]
        resultsShockList = ["SHCL"]
        # check for subtotals
        if "STLS" in HarObj.getHeaderArrayNames():
            self.appendSubtotals(HarObj, resultsDataHeaders, resultsSet, resultsShockComponents, resultsShockList)


        # Get the common header needed to extract the variable
        varNames=   HarObj["VARS"]
        varDims =   HarObj["VCNA"]
        varSetPtr = HarObj["VCAR"]
        varLabel  = HarObj["VCLB"]
        varSizeEnd = HarObj["ORND"]
        varSizeExo = HarObj["OREX"]
        varExoList = HarObj["OREL"]
        cumResCom = HarObj["CMND"]
        cumResPtr = HarObj["PCUM"]
        shockPtr  = HarObj["PSHK"]
        shockVal  = HarObj["SHOC"]


        varLower=[name.strip().lower() for name in varNames.array]
        if not extractList: extractList=varLower
        useVars=[item.lower() for item in extractList]
        useDict=dict(zip(useVars,useVars))

        self.varTypeDict=dict(zip(varLower,HarObj["VCTP"].array))
        #prepare the different results. By default add cumulative. If subtatoals are present append them to the lists

        self._variables=[name.strip() for name in varNames.array.tolist()]
        varSetDict={}
        nvar=len(self._variables)

        nexoListUsed=0
        nShkListUsed=0
        setPos=0

        self.variableDict={}

        #extract the variables. The resulting headers do not distinguish between endo and exo
        for i in range(0,nvar):
            useIt=varLower[i] in useDict
            setPos=self.generateSetDictEntry(i, resultsSet, setNames, setPos, varDims, varSetDict, varSetPtr)

            outDataList = self.assembleVariableData(HarObj, cumResCom, cumResPtr, i, nexoListUsed,
                                                                 resultsDataHeaders, resultsShockComponents,
                                                                 resultsShockList, shockPtr, shockVal, varExoList,
                                                                 varSizeEnd, varSizeExo, generateData=useIt)

            if useIt: self.reshapeAndAdd(i, outDataList, varLabel, varSetDict)
            nexo = varSizeExo.array[i, 0]
            nendo = varSizeEnd.array[i, 0]
            if nexo != 0 and nendo != 0: nexoListUsed += nexo


        # adjust available information to extracted data
        self._variables = [name.strip() for name in varNames.array.tolist() if name.strip().lower() in useDict]

        self.varTypeDict = { key:val for key,val in  self.varTypeDict.items() if key.lower() in useDict}

    def generateSetHeaders(self, HarObj, setNames):
        setSizes = HarObj["SSZ"]
        setLabels = HarObj["STLB"]
        setElStat = HarObj["ELST"]
        setEls = HarObj["STEL"]
        setElPtr = HarObj["ELAD"]
        self._sets = [name.strip().lower() for name in setNames.array.tolist()]
        nsets = len(self._sets)
        self.setHeaders = {}
        for i in range(0, nsets):
            if setElStat.array[i] == 'k':
                start = (setElPtr.array[i, 0] - 1)
                end = start + setSizes.array[i, 0]
                self.setHeaders[self._sets[i].strip().lower()] = HeaderArrayObj.SetHeaderFromData(self._sets[i],
                                                                                                  setEls.array[
                                                                                                  start:end],
                                                                                                  setLabels.array[i])

    def reshapeAndAdd(self, i, outDataList, varLabel, varSetDict):
        flatData = np.concatenate(outDataList)
        varSets = [thisSet.strip() for thisSet in varSetDict[self._variables[i].strip().lower()]]
        simSizes = tuple(
            [self.getSet(thisSet).array.shape[0] for thisSet in varSetDict[self._variables[i].strip().lower()]])
        setElDict = {}
        for myset in varSets:
            setElDict[myset.strip().lower()] = self.getSet(myset).array.tolist()
        finalData = flatData.reshape(simSizes,order="F").astype(np.float32)
        # create headers for all variables
        self.variableDict[self._variables[i].strip().lower()] = \
            HeaderArrayObj.HeaderArrayFromData(finalData,self._variables[i].strip()[0:12],varLabel.array[i][0:70], varSets, setElDict)

    def assembleVariableData(self, HarObj, cumResCom, cumResPtr, iVar, nexoListUsed, resultsDataHeaders,
                             resultsShockComponents, resultsShockList, shockPtr, shockVal, varExoList, varSizeEnd,
                             varSizeExo, generateData=True):
        nexo = varSizeExo.array[iVar, 0]
        nendo = varSizeEnd.array[iVar, 0]
        # Assemble the data into a vector (subtotals are appended to the list as they are in the results* Lists)
        outDataList = []
        if not generateData: return outDataList

        for DataHead, ShockComHead, ShockListHead in zip(resultsDataHeaders, resultsShockComponents, resultsShockList):
            cumRes = HarObj[DataHead]
            shockCom = HarObj[ShockComHead]
            shockList = HarObj[ShockListHead]
            start = cumResPtr.array[iVar, 0] - 1
            end = start + cumResCom.array[iVar, 0]
            Data = np.asfortranarray(cumRes.array[start:end, 0])
            nshk = shockCom.array[iVar, 0]
            nShkListUsed=0
            for prevVar in range(0,iVar):
                prevNEndo=varSizeEnd.array[prevVar, 0]
                prevNExo=varSizeExo.array[prevVar, 0]
                prevNShocked=shockCom.array[prevVar, 0]
                if prevNShocked != prevNExo or prevNEndo != 0: nShkListUsed+=prevNShocked

            if nexo != 0 and nendo != 0:# partially exo
                insertMask = []
                for j in range(nexoListUsed, nexoListUsed + nexo):
                    insertMask.append(varExoList.array[j, 0] - (j - nexoListUsed + 1))

                flatData = np.insert(Data, insertMask, 0)

                self.insertShocks(flatData, iVar, nshk, nexo, shockList, shockPtr, shockVal, nShkListUsed)
            elif nendo != 0: #fully endo
                flatData = Data
            else:  # fully exo
                flatData = np.zeros(nexo)
                self.insertShocks(flatData, iVar, nshk, nexo, shockList, shockPtr, shockVal, nShkListUsed)
            outDataList.append(flatData)
        return outDataList

    def generateSetDictEntry(self, i, resultsSet, setNames, setPos, varDims, varSetDict, varSetPtr):
        ndim = varDims.array[i, 0]
        if ndim > 0:
            varSetDict[self._variables[i].strip().lower()] = setNames.array[[j - 1 for j in varSetPtr.array[setPos:setPos + ndim, 0]]].tolist()
            varSetDict[self._variables[i].strip().lower()] = [name.strip() for name in varSetDict[self._variables[i].strip().lower()]]
            setPos += ndim
        else:
            varSetDict[self._variables[i].strip().lower()] = []
        if len(resultsSet) > 1: varSetDict[self._variables[i].strip().lower()].append("#RESULTS")
        return setPos

    def appendSubtotals(self, HarObj, resultsDataHeaders, resultsSet, resultsShockComponents, resultsShockList):
        nresults = HarObj["STLS"].array.flatten()[0]
        for i in range(1, nresults + 1):
            resultsDataHeaders.append("%03iS" % i)
            resultsShockComponents.append("%03iC" % i)
            resultsShockList.append("%03iL" % i)
            #resultsSet.append("Subtotal%03i" % i)
        description = ["Cumulative Results"]
        description.extend(HarObj["STDS"].array.tolist())
        for name in HarObj["STDS"].array:
            resultsSet.append(str(name) if len(name) <= 12 else str(name)[0:12])
        self._sets.append("#results")
        self.setHeaders["#results"] = HeaderArrayObj.SetHeaderFromData("#RESULTS", np.array(resultsSet), "Cumlative and Subtotal elements")

    @staticmethod
    def insertShocks(flatData, i, nshk, nexo, shockList, shockPtr, shockVal,nShkListUsed):
        if nshk > 0:
            start = shockPtr.array[i, 0] - 1
            if nshk == nexo:
                flatData[0:nexo]=shockVal.array[start:start+nshk,0]
            else:
                flatData[shockList.array[nShkListUsed:nShkListUsed+nshk,0]-1]=shockVal.array[start:start+nshk, 0]

