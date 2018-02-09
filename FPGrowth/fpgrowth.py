from numpy import *

class TreeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self,numOccur):
        self.count += numOccur

    def disp(self,ind = 1):
        print(" " * ind,self.name," ",self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def createTree(dataSet, minsupport = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
    headerTable = {k:v for k,v in headerTable.items() if v >= minsupport}
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:   return None,None
    for item in headerTable:
        headerTable[item] = [headerTable[item],None]
    root = TreeNode('Null set',1,None)
    for trans,count in dataSet.items():
        tempdict = {}
        for item in trans:
            if item in freqItemSet:
                tempdict[item] = headerTable[item][0]
        if len(tempdict) > 0:
            orderedItems = [v[0] for v in sorted(tempdict.items(),key = lambda p : p[1],reverse = True)]
            updateTree(orderedItems,root,headerTable,count)
    return root,headerTable

def updateTree(orderedItems,root,headerTable,count):
    if orderedItems[0] in root.children:
        root.children[orderedItems[0]].inc(count)
    else:
        root.children[orderedItems[0]] = TreeNode(orderedItems[0],count,root)
        if headerTable[orderedItems[0]][1] == None:
            headerTable[orderedItems[0]][1] = root.children[orderedItems[0]]
        else:
            updateHeader(headerTable[orderedItems[0]][1],root.children[orderedItems[0]])
    if len(orderedItems) > 1:
        updateTree(orderedItems[1::],root.children[orderedItems[0]],headerTable,count)

def updateHeader(nodeToTest,targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def ascendTree(treeNode,prefix):
    if treeNode.parent != None:
        prefix.append(treeNode.name)
        ascendTree(treeNode.parent,prefix)

def findPrefixPath(basePat,treeNode):
    condPats = {}
    while treeNode != None:
        prefix = []
        ascendTree(treeNode,prefix)
        if len(prefix) > 1: condPats[frozenset(prefix[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree,headerTable,minSup,prefix,freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(),key = lambda p : p[1][0])]
    for basePat in bigL:
        newFreqSet = prefix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1])
        myCondTree,myHead = createTree(condPattBases,minSup)
        if myHead != None:
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)