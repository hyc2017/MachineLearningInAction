from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMat,dim,threshVal,threshIneq):
    retArr = ones((shape(dataMat)[0],1))
    if threshIneq == 'lt':
        retArr[dataMat[:,dim] <= threshVal] = -1.0
    else:
        retArr[dataMat[:,dim] > threshVal] = -1.0
    return retArr

def buildStump(dataMat,labelMat,D):
    dataMat = mat(dataMat);labelMat = mat(labelMat).transpose()
    m,n = shape(dataMat)
    numsteps = 10.0;bestStump = {};bestEst = mat(zeros((m,1)))
    minError = inf

    for i in range(n):
        minVal = dataMat[:,i].min();maxVal = dataMat[:,i].max()
        stepSize = (maxVal - minVal) / numsteps
        for j in range(-1,int(numsteps) + 1):
            for inequal in ['lt','gt']:
                threshVal = minVal + float(j) * stepSize
                predictVals = stumpClassify(dataMat,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestEst = predictVals
                    bestStump['dim'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['threshIneq'] = inequal

    return bestStump,minError,bestEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    m = shape(dataArr)[0]
    D = mat(ones((m,1)) / m)
    weakClassArr = []
    aggClassExt = mat(zeros((m,1)))
    for it in range(numIt):
        stump,error,est = buildStump(dataArr,classLabels,D)
        #print("D:",D)
        alpha =  float(0.5*log((1.0 - error)/max(error,1e-16)))
        stump['alpha'] = alpha
        weakClassArr.append(stump)
        #print('classEst:',est.T)
        expon = multiply(-1 * alpha * mat(classLabels).T,est)
        D = multiply(D,exp(expon))
        D = D / D.sum()
        aggClassExt += alpha * est
        #print('aggClassExt:',aggClassExt.T)
        aggErrors = multiply(sign(aggClassExt) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum() / m
        print('total error:',errorRate)
        if errorRate == 0.0:break;
    return weakClassArr

def adaClassify(dataToClassify,classifierArr):
    dataMat = mat(dataToClassify)
    m = shape(dataMat)[0]
    classEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        predict = stumpClassify(dataMat,classifierArr[i]['dim'],classifierArr[i]['threshVal'],classifierArr[i]['threshIneq'])
        classEst += classifierArr[i]['alpha'] * predict
        #print('aggClassEst : ',classEst)
    return sign(classEst)

