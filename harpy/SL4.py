from __future__ import print_function, absolute_import
from .HAR import HAR
from .Header import Header
import numpy as np


class SL4(object):

    def __init__(self, fname):
        self.HARobj=HAR(fname,'r')
        self.decode_SL4()
        self.HARobj.f.f.close()

    @property
    def variables(self):
        return self._variables
    @variables.setter
    def variables(self):
        raise Exception ("Variables in SL4 class cannot be assigned")

    @property
    def sets(self):
        return self.sets
    @sets.setter
    def sets(self):
        raise Exception ("Sets in SL4 class cannot be assigned")

    def getSet(self,name):
        if not name.strip() in self._sets:
            raise Exception ("Could not find set '"+name+"' in the SL4 file. Please check the set list")
        return self.setHeaders[name.strip()]

    def getVariable(self,name):
        if not name.strip() in self._variables:
            raise Exception("Could not find variable '" + name + "' in the SL4 file. Please check the variable list")
        return self.variableDict[name]

    def decode_SL4(self):

        #collect the set information. Not sure whether intertemporals work yet
        setNames =   self.HARobj.getHeader("STNM")
        setSizes =   self.HARobj.getHeader("SSZ ")
        setLabels  = self.HARobj.getHeader("STLB")
        setElStat =  self.HARobj.getHeader("ELST")
        setEls =     self.HARobj.getHeader("STEL")
        setElPtr =   self.HARobj.getHeader("ELAD")

        self._sets=[name.strip() for name in setNames.DataObj.tolist()]
        nsets=len(self._sets)
        self.setHeaders={}
        for i in range(0,nsets):
            if setElStat.DataObj[i] == 'k':
                start=(setElPtr.DataObj[i,0]-1)
                end=start+setSizes.DataObj[i,0]
                self.setHeaders[self._sets[i]]=Header.SetHeaderFromData("S%03i" % i, setEls.DataObj[start:end], self._sets[i], setLabels.DataObj[i])

        # Get the common header needed to extract the variable
        varNames=  self.HARobj.getHeader("VARS")
        varDims =  self.HARobj.getHeader("VCNA")
        varSetPtr =  self.HARobj.getHeader("VCAR")
        varLabel= self.HARobj.getHeader("VCLB")
        varSizeEnd = self.HARobj.getHeader("ORND")
        varSizeExo = self.HARobj.getHeader("OREX")
        varExoList = self.HARobj.getHeader("OREL")
        cumResCom = self.HARobj.getHeader("CMND")
        cumResPtr = self.HARobj.getHeader("PCUM")
        shockPtr  = self.HARobj.getHeader("PSHK")
        shockVal  = self.HARobj.getHeader("SHOC")

        #prepare the different results. By default add cumulative. If subtatoals are present append them to the lists
        resultsSet=["Cumulative"]
        resultsDataHeaders=["CUMS"]
        resultsShockComponents = ["SHCK"]
        resultsShockList = ["SHCL"]
        # check for subtotals
        if "STLS" in self.HARobj.HeaderNames():
            self.appendSubtotals(resultsDataHeaders, resultsSet, resultsShockComponents, resultsShockList)


        self._variables=[name.strip() for name in varNames.DataObj.tolist()]
        varSetDict={}
        nvar=len(self._variables)
        nexoUsed=0; setPos=0
        self.variableDict={}

        #extract the variables. The resulting headers do not distinguish between endo and exo
        for i in range(0,nvar):
            ndim=varDims.DataObj[i,0]
            if ndim>0:
                varSetDict[self._variables[i]]=setNames.DataObj[[j - 1 for j in varSetPtr.DataObj[setPos:setPos + ndim, 0]]].tolist()
                setPos+=ndim
            else:
                varSetDict[self._variables[i]]=[]
            if len(resultsSet)>1:varSetDict[self._variables[i]].append("#RESULTS")

            nexo = varSizeExo.DataObj[i, 0]
            nendo = varSizeEnd.DataObj[i, 0]

            # Assemble the data into a vector (subtotals are appended to the list as they are in the results* Lists)
            outDataList=[]
            for DataHead,ShockComHead,ShockListHead in zip(resultsDataHeaders, resultsShockComponents, resultsShockList):
                cumRes=self.HARobj.getHeader(DataHead)
                shockCom = self.HARobj.getHeader(ShockComHead)
                shockList = self.HARobj.getHeader(ShockListHead)

                start = cumResPtr.DataObj[i, 0] - 1
                end = start + cumResCom.DataObj[i, 0]
                Data = cumRes.DataObj[start:end, 0]

                nshk = shockCom.DataObj[i, 0]
                if nexo != 0 and nendo != 0:
                    insertMask=[]
                    for j in range(nexoUsed, nexoUsed+ nexo):
                        insertMask.append(varExoList.DataObj[j,0]-(j-nexoUsed+1))
                    nexoUsed+= nexo

                    flatData=np.insert(Data,insertMask,0)

                    self.insertShocks(flatData, i, nshk, nexo, shockList, shockPtr, shockVal)
                elif nendo != 0:
                    flatData=Data
                else:
                    flatData=np.zeros((nexo))
                    self.insertShocks(flatData, i, nshk, nexo, shockList, shockPtr, shockVal)
                outDataList.append(flatData)

            flatData=np.concatenate(outDataList)

            varSets=[thisSet for thisSet in varSetDict[self._variables[i]]]

            simSizes=tuple([self.setHeaders[thisSet].DataObj.shape[0] for thisSet in varSetDict[self._variables[i]]])
            setElDict={}
            for myset in varSets:
                setElDict[myset]=self.setHeaders[myset].DataObj.tolist()
            finalData=flatData.reshape(simSizes,order='F').astype(np.float32)

            #create headers for all variables
            name="%04i"%(i+1)
            self.variableDict[self._variables[i]]=Header.HeaderFromData(name, finalData, varLabel.DataObj[i][0:70], self._variables[i].strip()[0:12], varSets, setElDict)

    def appendSubtotals(self, resultsDataHeaders, resultsSet, resultsShockComponents, resultsShockList):
        nresults = self.HARobj.getHeader("STLS").DataObj[0, 0]
        for i in range(1, nresults + 1):
            resultsDataHeaders.append("%03iS" % i)
            resultsShockComponents.append("%03iC" % i)
            resultsShockList.append("%03iL" % i)
            resultsSet.append("Subtotal%03i" % i)
        description = ["Cumulative Results"]
        description.extend(self.HARobj.getHeader("STDS").DataObj.tolist())
        self.SimDescHead = Header.HeaderFromData("DESC", np.array(description),
                                                 "Content of results, i.e. cumlative and subtotal content", "",
                                                 ["#RESULTS"], [resultsSet])
        self._sets.append("#RESULTS")
        self.setHeaders["#RESULTS"] = Header.SetHeaderFromData("R000", np.array(resultsSet), "#RESULTS",
                                                               "Cumlative and Subtotal elements")

    def insertShocks(self, flatData, i, nshk, nexo, shockList, shockPtr, shockVal):
        if nshk > 0:
            start = shockPtr.DataObj[i, 0] - 1
            for j in range(0, nshk):
                shkInd = start + j
                if nshk == nexo:
                    varInd=j
                else:
                    varInd = shockList.DataObj[j, 0] - 1
                flatData[varInd] = shockVal.DataObj[shkInd, 0]

