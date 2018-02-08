from numpy import *

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 = []
    for line in dataSet:
        for item in line:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))

def scanD(D,Ck,minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key)
            supportData[key] = support
    return retList,supportData


def aprioriGen(Ck,k):
    m = len(Ck)
    retList = []
    for i in range(m):
        for j in range(i + 1,m):
            L1 = list(Ck[i])[:k - 2];L2 = list(Ck[j])[:k - 2]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append(Ck[i] | Ck[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while(len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1  
    return L,supportData

def generateRules(L,supportData,minConf = 0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqset in L[i]:
            H1 = [frozenset([item]) for item in freqset]
            if i != 1:
                rulesFromConseq(freqset,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqset,H1,supportData,bigRuleList,minConf)
    return bigRuleList

def calcConf(freqset,H1,supportData,bigRuleList,minConf):
    prunedH = []
    for conseq in H1:
        conf = supportData[freqset] / supportData[freqset - conseq]
        if conf >= minConf:
            print(freqset - conseq,'-->',conseq,',conf:',conf)
            bigRuleList.append((freqset - conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqset,H1,supportData,bigRuleList,minConf):
    m = len(H1[0])
    if len(freqset) > m + 1:
        Hmp1 = aprioriGen(H1,m + 1)
        prunedH = calcConf(freqset,Hmp1,supportData,bigRuleList,minConf)
        if len(prunedH) > 1:
            rulesFromConseq(freqset,prunedH,supportData,bigRuleList,minConf)

