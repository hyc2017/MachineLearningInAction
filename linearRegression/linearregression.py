from numpy import *

def loadDataSet(filename):
    featNum = len(open(filename).readline().split()) - 1
    dataMat = [];labelMat = []
    for line in open(filename).readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(featNum):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#linear regression
def standRerges(xArr,yArr):
    dataMat = mat(xArr);labelMat = mat(yArr).T
    XTX = dataMat.T * dataMat
    if linalg.det(XTX) == 0.0:
        print("the dataMatrix is singular,can't get its inverse"); return;
    W = XTX.I * (dataMat.T * labelMat)
    return W

#locally weighted linear regression
def lwlr(testpoint,dataArr,labelArr,k = 1.0):
    xMat = mat(dataArr);yMat = mat(labelArr).T
    m = shape(xMat)[0]
    weight = mat(eye((m)))

    for i in range(m):
        diffMat = testpoint - xMat[i,:]
        weight[i,i] = exp(diffMat * diffMat.T /(-2*k**2))
    XTX = xMat.T *  (weight * xMat)
    if linalg.det(XTX) == 0.0:
        print("the dataMtrix is singular,can't get its inverse");return;
    ws = XTX.I * (xMat.T * (weight * yMat))
    return testpoint * ws

def lwlrTest(testArr,dataArr,labelArr,k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],dataArr,labelArr,k)
    return yHat

def rssError(yMat,yHat):
    return ((yMat - yHat) ** 2).sum()

def ridgeRegres(xMat,yMat,lam = 0.2):
    XTX = xMat.T * xMat + eye(shape(xMat)[1]) * lam
    if linalg.det(XTX) == 0.0:
        print("matrix is singular,can't get its reverse");return;
    ws = XTX.I * (xMat.T * yMat)
    return ws   

def ridgeTest(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    ymean = mean(yMat,0)
    yMat = yMat - ymean
    xmean = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xmean) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1));wsTest = ws.copy();wsBest = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestErr = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestErr:
                    wsBest = wsTest
                    lowestErr = rssE
        ws = wsBest.copy()
        returnMat[i,:] = ws.T
    return returnMat