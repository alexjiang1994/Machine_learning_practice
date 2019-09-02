'''
-*- coding: utf-8 -*-
@Author  : Alex
@Time    : 2019/8/31 10:11
@Software: PyCharm
@File    : trees.py
'''
import math


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEnteries = len(dataSet)
    labelsCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelsCounts[currentLabel] = labelsCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for k in labelsCounts.values():
        prob = k / numEnteries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, value, axis):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            vector = featVec[:axis]
            vector.extend(featVec[axis+1:])
            retDataSet.append(vector)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = -1; bestInfoGain = 0.0
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntro = 0.0
        for value in uniqueVals: # 求按照该特征划分后的条件熵
            subDataSet =  splitDataSet(dataSet, value, i)
            prob = len(subDataSet) / len(dataSet)
            newEntro += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntro
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sorted_classCount = sorted(classCount.items(), key=lambda k:k[1])
    return sorted_classCount[-1][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 所有纪录的值相同，停止分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 该分类下，只有一个特征，停止分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[0:bestFeat] + labels[bestFeat+1:]
        myTree[bestFeatLabel][value] = createTree(\
            splitDataSet(dataSet, value, bestFeat), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fw = open(filename, 'r')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    myTree = createTree(dataSet, labels)
    print(myTree)
