from numpy import * 
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for line in dataSet:
        vocabSet  = vocabSet | set(line)
    return list(vocabSet)

def setOfWordsToVec(vocabSet,inputData):
    returnVec = [0] * len(vocabSet)
    for word in inputData:
        if word in vocabSet:
            returnVec[vocabSet.index(word)] = 1
        else:
            print("the word %s is not in my vocabulary!!" % word)
    return returnVec

def bagOfWordsToVecMN(vocabSet,inputData):
    returnVec = [0] * len(vocabSet)
    for word in inputData:
        if word in vocabSet:
            returnVec[vocabSet.index(word)] += 1
        else:
            print("the word %s is not in my vocabulary!!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    abusive = sum(trainCategory) / len(trainCategory)
    #避免单词个数为0，导致概率为0
    P0 = ones(numWords); P1 = ones(numWords)
    P0Dom = 2.0; P1Dom = 2.0

    for i in range(numDocs):
        if trainCategory[i] == 0:
            P0 += trainMatrix[i]
            P0Dom += sum(trainMatrix[i])
        else:
            P1 += trainMatrix[i]
            P1Dom += sum(trainMatrix[i])
    #将p(A) * p(B) 转化为log(p(A)) + log(p(B)),避免下溢出或者四舍五入导致答案不正确
    P0Vec = log(P0 / P0Dom)
    P1Vec = log(P1 / P1Dom)
    return P0Vec,P1Vec,abusive

#对文档进行分类
def classify(vec2Classify,P0Vec,P1Vec,pClass1):
    p1 = sum(vec2Classify * P1Vec) + log(pClass1)
    p0 = sum(vec2Classify * P0Vec) + log(1- pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#测试方法
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWordsToVec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWordsToVec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classify(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWordsToVec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classify(thisDoc,p0V,p1V,pAb))
