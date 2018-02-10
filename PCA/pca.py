from numpy import *

def  loadDataSet(filename,delim='\t'):
    fr = open(filename)
    Arr = [line.strip().split(delim) for line in fr.readlines()]
    dataMat = [list(map(float,line)) for line in Arr]
    return mat(dataMat)

def pca(dataMat,topNfeat = 9999):
    meanVal = mean(dataMat,axis = 0)
    subMean = dataMat - meanVal
    covMat = cov(subMean,rowvar = 0)
    eigVals,eigVec = linalg.eig(mat(covMat))
    eigValInds = argsort(eigVals)
    eigValInds = eigValInds[:-(topNfeat + 1):-1]
    redEigVals = eigVec[:,eigValInds]
    lowDMat = subMean * redEigVals
    reconMat = (lowDMat * redEigVals.T) + meanVal
    return lowDMat,reconMat