from math import log
import operator

#通过香农熵表示数据集的混乱程度
#公式为p(x) * log(p(x),2),其中x表示数据集中的每个类别
def calcShannonEnt(dataSet):
    m = len(dataSet)
    labelCount = {}

    for line in dataSet:
        currentLabel = line[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCount:
        prob = labelCount[key] / m
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#根据数据集的对应列的对应值划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for line in dataSet:
        if line[axis] == value:
            row = line[:axis]
            row.extend(line[axis+1:])
            retDataSet.append(row)
    return retDataSet

#遍历当前数据集所有特征，找出使香农熵下降最多的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    baseInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featlist = [line[i]  for line in dataSet]
        uniqueVals =  set(featlist)
        entropy = 0.0
        for value in uniqueVals:
            data = splitDataSet(dataSet,i,value)
            shannon = calcShannonEnt(data)
            prob = len(data) / float(len(dataSet))
            entropy += prob * shannon
        if(baseEntropy - entropy > baseInfoGain):
            bestFeature = i
            baseInfoGain = baseEntropy - entropy
    return bestFeature

#多数表决
def majorityVote(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #根据count对map进行排序
    sortedVote = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedVote[0][0]


#构建决策树
def createTree(dataSet,labels):
    classlist = [line[-1] for line in dataSet]
    if(len(classlist) == classlist.count(classlist[0])):
        return classlist[0]
    if(len(dataSet[0]) == 1):
        return majorityVote(classlist)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featVals = [line[bestFeat] for line in dataSet]
    uniqueVals = set(featVals)
    for i in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][i] = createTree(splitDataSet(dataSet,bestFeat,i),subLabels)
    return myTree

#利用决策树进行分类
def classify(inputTree,labels,inx):
    #注意，这里与书中代码不同,需要将keys集合转化为list
    firststr = list(inputTree.keys())[0]
    seconddict = inputTree[firststr]
    featindex = labels.index(firststr)
    for key in seconddict.keys():
        if inx[featindex] == key:
            if type(seconddict[key]).__name__ == 'dict':
                classlabel = classify(seconddict[key],labels,inx)
            else:
                classlabel = seconddict[key]
    return classlabel

