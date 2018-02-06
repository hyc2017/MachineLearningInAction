from numpy import *

def loadDataSet(filename):
    dataMat = []
    for line in open(filename).readlines():
        strs = line.strip().split('\t')
        ftline = list(map(float,strs))
        dataMat.append(ftline)
    return dataMat

def binSplitDataSet(dataSet,ind,val):
    mat0 = dataSet[nonzero(dataSet[:,ind] > val)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,ind] <= val)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return mean(dataSet[:,-1])


def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#leafType,function to bulid tree
#errType,lossfunction
#ops,arguments for buliding tree
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None : return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


def chooseBestSplit(dataSet,leafType,errType,ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    BestS = inf; bestind = 0; bestval = 0
    for featindex in range(n - 1):
        for splitval in set(dataSet[:,featindex].flatten().A[0]):
            left,right = binSplitDataSet(dataSet,featindex,splitval)
            if (shape(left)[0] < tolN) or (shape(right)[0] < tolN):continue
            newS = errType(left) + errType(right)
            if newS < BestS:
                bestind = featindex
                bestval = splitval
                BestS = newS
    if S - BestS < tolS:    return None,leafType(dataSet)
    left,right = binSplitDataSet(dataSet,bestind,bestval)
    if (shape(left)[0] < tolN) or (shape(right)[0] < tolN): return None,leafType(dataSet)
    return bestind,bestval

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):   tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):    tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree,testData):
    if shape(testData)[0] == 0: return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        left,right = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):    tree['left'] = prune(tree['left'],left)
    if isTree(tree['right']):   tree['right'] = prune(tree['right'],right)
    if not isTree(tree['left']) and not isTree(tree['right']):
        left,right = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(left[:,-1] - tree['left'],2)) + sum(power(right[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:   return tree
    else:   return tree

#get width of the tree,more convenient to figure out the outcome of pruning
def getWidth(tree):
    width = 0
    if isTree(tree):
        width += getWidth(tree['left'])
        width += getWidth(tree['right'])
    else:
        return 1
    return width

#here are three methods to help build model tree
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)));Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1];Y = dataSet[:,-1]
    XTX = X.T * X
    if linalg.det(XTX) == 0.0:
        raise NameError('sigular martix,can not get its inversed')
    ws = XTX.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(yHat - Y,2))