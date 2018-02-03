import random
from numpy import *

def loadDataSet(filename):
    dataMat = [];labelMat = []
    file = open(filename)

    for line in file.readlines():
        line = line.strip().split('\t')
        dataMat.append([float(line[0]),float(line[1])])
        labelMat.append(float(line[2]))

    return dataMat,labelMat 

def randSelect(i,m):
    j =  i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

#简化版smo
def smoSimple(dataMat,labelMat,C,toler,maxiter):
    dataMat = mat(dataMat);labelMat = mat(labelMat).transpose()

    b = 0;m,n = shape(dataMat)
    alphas = mat(zeros([m,1]))
    iter = 0

    while(iter < maxiter):
        alphapairchanged = 0
        for i in range(m):
            fxi = float(multiply(alphas,labelMat).T * (dataMat * dataMat[i,:].T)) + b
            Ei = fxi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and alphas[i] < C) or ((labelMat[i] * Ei > toler) and alphas[i] > 0):
                j = randSelect(i,m)
                fxj = float(multiply(alphas,labelMat).T * (dataMat * dataMat[j,:].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIhold = alphas[i].copy();alphaJhold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[i] + alphas[j])
                if(L == H): print("L == H");continue
                eta = 2.0 * dataMat[i,:] * dataMat[j,:].T - dataMat[i,:] * dataMat[i,:].T  - dataMat[j,:] * dataMat[j,:].T
                if(eta >= 0): print("eta >= 0");continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if(abs(alphas[j] - alphaJhold) < 0.00001): print("j not moving enough");continue
                alphas[i] += labelMat[i] * labelMat[j] * (alphaJhold - alphas[j])
                b1 = b -Ei - labelMat[i] * dataMat[i,:] * dataMat[i,:].T * (alphas[i] - alphaIhold) - labelMat[j] * dataMat[i,:] * dataMat[j,:].T * (alphas[j] - alphaJhold)
                b2 = b -Ej - labelMat[i] * dataMat[i,:] * dataMat[j,:].T * (alphas[i] - alphaIhold) - labelMat[j] * dataMat[j,:] * dataMat[j,:].T * (alphas[j] - alphaJhold)  
                if(0 < alphas[i] and alphas[i] < C): b = b1
                elif(0 < alphas[j] and alphas[j] < C): b = b2
                else: b = (b1 + b2) / 2.0
                alphapairchanged += 1
                print("iter: %d i : %d,pair changed:%d" % (iter,i,alphapairchanged))

        if(alphapairchanged == 0):iter += 1
        else:   iter = 0
        print("iteration number: %d" % iter)
    return b,alphas



class optStruct:
    def __init__(self,dataMat,labelMat,C,toler,kTup):
        self.X = dataMat
        self.labelMat = labelMat
        self.C = C
        self.toler = toler
        self.m = shape(dataMat)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

def calEk(oS,k):
    fxk = float(multiply(oS.alphas,oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fxk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    Ej = 0;maxK = -1;maxDelta = 0
    oS.eCache[i] = [1,Ei]
    validEcachList = nonzero(oS.eCache[:,0].A)[0]
    if len(validEcachList) > 1:
        for k in validEcachList:
            if k == i: continue
            Ek = calEk(oS,k)
            delta = abs(Ei - Ek)
            if delta > maxDelta:
                maxDelta = delta; maxK = k; Ej = Ek
        return maxK,Ej
    else:
        k = randSelect(i,oS.m)
        Ek = calEk(oS,k)
        return k,Ek

def updateEk(oS,k):
    Ek = calEk(oS,k)
    oS.eCache[k] = [1,Ek]

def innerL(i,oS):
    Ei = calEk(oS,i)
    if  ((oS.alphas[i] < oS.C) and (oS.labelMat[i] * Ei < -oS.toler)) or ((oS.alphas[i] > 0) and (oS.labelMat[i] * Ei > oS.toler)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIhold = oS.alphas[i].copy();alphaJhold = oS.alphas[j].copy()
        if(oS.labelMat[i] * oS.labelMat[j] < 0):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C,oS.alphas[i] + oS.alphas[j])
        eta = 2 * oS.K[i,j]- oS.K[i,i] - oS.K[j,j]
        if eta >= 0 : print("eta >= 0");return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if abs(oS.alphas[j] - alphaJhold < 0.00001): print("j not moving enough!"); return 0
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJhold - oS.alphas[j])
        updateEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i] * oS.K[i,i] * (oS.alphas[i] - alphaIhold) - oS.labelMat[j] * oS.K[j,i] * (oS.alphas[j] - alphaJhold)
        b2 = oS.b - Ej - oS.labelMat[i] * oS.K[i,j] * (oS.alphas[i] - alphaIhold) - oS.labelMat[j] * oS.K[j,j] * (oS.alphas[j] - alphaJhold)
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:    oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:  oS.b = b2
        else:   oS.b = (b1 + b2) / 2.0
        return 1
    else: return 0

def smoP(dataMat,labelMat,C,toler,maxIter,kTup = ('lin',0)):
    oS = optStruct(mat(dataMat),mat(labelMat).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and (alphaPairsChanged > 0) or entireSet:
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            print("fullSet, iter: %d i : %d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundsI = nonzero((oS.alphas.A > 0 ) * (oS.alphas.A < oS.C))[0]
            for i  in nonBoundsI:
                alphaPairsChanged += innerL(i,oS)
                print("nonBound, iter: %d, i : %d, pais changed : %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet : entireSet  = False
        elif (alphaPairsChanged == 0) : entireSet = True

        print("iteration number : %d" % iter)
    return oS.b,oS.alphas

def kernelTrans(X,A,kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin' : K = X * A.T
    elif kTup[0] == 'rbf':
        for k in range(m):
            deltaRow = X[k,:] - A
            K[k] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else: raise NameError('Houston we have a Problem -- That Kernel is not recogonized!')
    return K

def testRbf(k1 = 0.8):
    dataArr,labelArr = loadDataSet('testSetRbf.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = mat(dataArr);labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    svs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d support vectors' % shape(labelSV)[0])
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs,dataMat[i,:],('rbf',k1))
        predictVal = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predictVal) != labelMat[i] : errorCount += 1
    print('training error rate is %f' % (float(errorCount) / m))

    dataMat,labelMat = loadDataSet('testSetRbf2.txt')
    dataMat = mat(dataMat);labelMat = mat(labelMat).transpose()
    m,n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs,dataMat[i,:],('rbf',k1))
        predictVal = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predictVal) != labelMat[i] : errorCount += 1
    print('test error rate is %f' % (float(errorCount) / m))


                




