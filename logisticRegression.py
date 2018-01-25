from numpy import *
def loadDataSet():
    fr = open('testSet.txt')
    dataMat = []; labelMat = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))

#梯度下降
def gradAscent(dataMatIn,classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n  = shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for i in range(maxCycles):
        predictVasls = sigmoid(dataMat * weights)
        errors = labelMat - predictVasls
        weights += alpha * dataMat.transpose() * errors
    return weights

#随机梯度下降
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    weights = ones(n)
    alpha = 0.001
    for i in range(m):
        predictI = sigmoid(dataMatrix[i] * weights)
        error = classLabels[i] - predictI
        weights += alpha * dataMatrix[i] * error
    return weights

#随机梯度下降改进版。主要在于a的动态调整和随机样本选取
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01
            randindex = int(random.uniform(0,len(dataIndex)))
            predict = sigmoid(sum(dataMatrix[randindex] * weights))
            error = classLabels[randindex] - predict
            weights += alpha * dataMatrix[randindex] * error
            del(dataIndex[randindex])
    return weights

#分类
def classify(inx,weights):
    prob = sigmoid(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0