'''
-*- coding: utf-8 -*-
@Author  : Alex
@Time    : 2019/8/28 10:14
@Software: PyCharm
@File    : knn.py
k-近邻算法
优点：精度高、对异常值不敏感、无数据输入假定
缺点：计算复杂度高、空间复杂度高
使用数据范围：数值型和标称型
'''

import numpy as np
import operator
import os

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 分类思想是最重要的，1.求距离 2.取最小的k个值 3.进行投票或者加权投票
def classify0(inX, dataSet, labels, k):
    #
    dataSetSize = dataSet.shape[0] # get num of sample
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = np.power(diffMat, 2)
    sqDistances = sqDiffMat.sum(axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistIndicies = distances.argsort() # Get the first k minimum's index
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda k:k[1], reverse=True)
    return sortedClassCount[0][0] # get the voted result


def classify1(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    Mat = np.tile(inX, (dataSetSize, 1)) - dataSet
    Mat = Mat ** 2
    Mat = Mat.sum(axis=1)
    Mat = Mat ** 0.5
    sortedDistIndices = Mat.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda k:k[1], reverse=True)
    return sortedClassCount[0][0]

def classify2(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    number_lines = len(arrayLines)
    returnMat = np.zeros((number_lines, 3))
    classLabelVector = []
    index = 0
    d = {'smallDoses':1, 'didntLike':0, "largeDoses":2}
    for line in arrayLines:
        line = line.strip().split('\t')
        returnMat[index, :] = line[0:3]
        classLabelVector.append(d[(line[-1])])
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals

    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingclassTest():
    hoRatio = 0.2
    filename = 'datingTestSet.txt'
    dataSet, labels = file2matrix(filename)
    normDataSet, ranges, minVals = autoNorm(dataSet)
    m = normDataSet.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        result = classify0(normDataSet[i], normDataSet[numTestVecs:], labels[numTestVecs:], 3)
        print("the classifier came back with: {}, the real answer is: {}".format(result, labels[i]))
        if result != labels[i]:
            errorCount += 1
    print("the total error rate is: {}".format(errorCount/float(numTestVecs)))
    print(errorCount)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percents = float(input("Percentage of time spent playing video games?"))
    miles = float(input("frequent flier miles earned per year?"))
    icecream = float(input("liters of ice cream consumed per year?"))
    dataSet, labels = file2matrix("datingTestSet.txt")
    normDataSet, ranges, minVals = autoNorm(dataSet)
    inX = (np.array([miles, percents, icecream]) - minVals) / ranges
    result = classify0(inX, normDataSet, labels, 3)
    print("You will probably like this person:{}".format(resultList[result]))


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')
    testFileList = os.listdir('digits/testDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        classNumStr = filenameStr.split("_")[0]
        hwLabels.append(int(classNumStr))
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % filenameStr)
    mTest = len(testFileList)
    errorCount = 0
    for i in range(mTest):
        filenameStr = testFileList[i]
        classNumStr = int(filenameStr.split("_")[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' %filenameStr)
        result = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        if result != classNumStr:
            errorCount += 1
        print("the classifier came back with: %d, the real answer is: %d" % (result, classNumStr))
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()