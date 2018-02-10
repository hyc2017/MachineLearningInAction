from numpy import *
from numpy import linalg as la

def loadExData():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    return 0.5 + 0.5 * corrcoef(inA,inB,rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0;simRate = 0.0

    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0:continue
        overLap = nonzero(logical_and(dataMat[:,item].A > 0,dataMat[:,j].A > 0))[0]
        if len(overLap) == 0:similarity = 0
        else:   similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        print('the similarity between %d and %d is %f' % (item,j,similarity))
        simRate += userRating * similarity
        simTotal += similarity
    if simTotal == 0: return 0
    else: return simRate / simTotal

def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0;simRate = 0.0

    U,Sigma,VT = la.svd(dataMat)
    Sig3 = mat(eye(3) * Sigma[:3])
    xformedItems = dataMat.T * U[:,:3] * Sig3.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or item == j: continue
        similarity = simMeas(xformedItems[j,:].T,xformedItems[item,:].T)
        print('the similarity between %d and %d is %f' % (item,j,similarity))
        simTotal += similarity
        simRate += similarity * userRating
    if simTotal == 0: return 0
    else: return simRate / simTotal


def recommend(dataMat,user,N = 3,simMeas = ecludSim,estMethod = standEst):
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0:print("you have rated everything!")

    itemScores = []
    for item in unratedItems:
        score = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,score))
    return sorted(itemScores,key = lambda p : p[1],reverse = True)[:N]

