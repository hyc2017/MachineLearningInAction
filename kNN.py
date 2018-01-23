from numpy import *
import operator

def classify(inx,dataSet,labels,k = 3):
    m = dataSet.shape[0]
    diffMat = tile(inx,(m,1)) - dataSet
    squareMat = diffMat ** 2
    squareDistance = squareMat.sum(axis = 1)
    distances = squareDistance ** 0.5
    indexArr = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[indexArr[i]]
        classCount[label] = classCount.get(label,0) + 1 
    sortedCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedCount[0][0]   

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    diffMat = dataSet - tile(minVals,(m,1))
    return diffMat / tile(ranges,(m,1))
    