from numpy import *

def loadDataSet(filename):
    dataMat = []
    for line in open(filename).readlines():
        strs = line.strip().split('\t')
        floatline = list(map(float,strs))
        dataMat.append(floatline)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA - vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for i in range(n):
        minVal = min(dataSet[:,i])
        rangeVal = float(max(dataSet[:,i]) - minVal)
        centroids[:,i] = minVal + rangeVal * random.rand(k,1)
    return centroids

def kmeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]
    centroids = createCent(dataSet,k)
    assessment = mat(zeros((m,2)))
    clusterchanged = True
    while clusterchanged:
        clusterchanged = False
        for i in range(m):
            bestdist = inf;bestk = 0
            for j in range(k):
                dist = distMeas(dataSet[i],centroids[j])
                if dist < bestdist:
                    bestdist = dist
                    bestk = j
            if assessment[i,0] != bestk:
                clusterchanged = True
                assessment[i,:] = bestk,bestdist
        for cent in range(k):
            ptsInClust = dataSet[nonzero(assessment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis = 0)
    return centroids,assessment

def biKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2
    while len(centList) < k:
        lowestErr = inf
        for cent in range(len(centList)):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0],:]
            centroids,assessment = kmeans(ptsInClust,2,distMeas)
            ssesplit = sum(assessment[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != cent)[0],1])
            print("errorSplit, and not split :",ssesplit,sseNotSplit)
            if ssesplit + sseNotSplit < lowestErr:
                lowestErr =  ssesplit + sseNotSplit
                bestCentToSplit = cent
                bestNewCentroids = centroids
                bestClusterAssment = assessment.copy()
        bestClusterAssment[nonzero(bestClusterAssment[:,0].A == 1)[0],0] = len(centList)
        bestClusterAssment[nonzero(bestClusterAssment[:,0].A == 0)[0],0] = bestCentToSplit
        print("the best cent to split is ",bestCentToSplit)
        print("len of bestClustAss is ",len(bestClusterAssment))
        centList[bestCentToSplit] = centroids[0,:].tolist()[0]
        centList.append(centroids[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClusterAssment
    return mat(centList),clusterAssment
