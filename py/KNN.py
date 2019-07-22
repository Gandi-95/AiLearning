import operator

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


def file2Matrix(filename):
    '''
    Desc:
        导入数据
    param:
        filename:数据文件路径
    return:
        数据矩阵 retureMat 、 数据类别 classLabelVector
    '''
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    # index = 0
    for index, line in enumerate(fr.readlines()):
        listFromLine = line.strip().split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int (listFromLine[-1]))
        # print(line)
    return returnMat, classLabelVector

def show(returnMat, classLabelVector):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:, 0], returnMat[:, 1], 15.0 * np.array(classLabelVector), 15.0 * np.array(classLabelVector))
    plt.show()

def autoNorm(dataSet):
    """
       Desc:
           归一化特征值，消除特征之间量级不同导致的影响
       parameter:
           dataSet: 数据集
       return:
           归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

       归一化公式：
           Y = (X-Xmin)/(Xmax-Xmin)
           其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    print("minVals:"+str(minVals)+"\nmaxVals:"+str(maxVals))
    # 极差
    ranges = maxVals - minVals
    #  m 表示数据的行数，即矩阵的第一维
    m = np.shape(dataSet)[0]
    # 通过np.tile 函数生成和dataSet同样大小的矩阵
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    dataSetSize = np.shape(dataSet)[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    # axis=1 将一个矩阵的每一行向量相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # argsort 从小到大排序并且提取对应的index
    sortedDistIndicies = np.argsort(distances)
    # print(sortedDistIndicies)
    classCount={}
    for i in range(k):
        voltelable = labels[sortedDistIndicies[i]]
        # 获取到lable保存到字典里，value++
        classCount[voltelable] = classCount.get(voltelable,0)+1
    # 排序并返回出现最多的那个类型
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    """
        Desc：
            对约会网站的测试方法，并将分类错误的数量和分类错误率打印出来
        Args：
            None
        Returns：
            None
    """
    returnMat, classLabelVector =file2Matrix("./data/knn/datingTestSet2.txt")
    # 归一化数据
    normMat, ranges, minVals = autoNorm(returnMat)
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    print('numTestVecs=', numTestVecs)
    for i in range(numTestVecs):
        result = classify0(normMat[i,:],normMat[numTestVecs:m],classLabelVector[numTestVecs:m],3)
        print(str(result)+":"+str(classLabelVector[i]))
        if (result != classLabelVector[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def img2vector(filename):
    returnVect = np.zeros((1,1024))
    with open(filename,'r',encoding='utf-8') as f:
        for i in range(32):
            lienStr = f.readline()
            for j in range(32):
                returnVect[0,i*32+j] = int(lienStr[j])
    return returnVect

def handwritingClassTest():
    hwlables = []
    trainFileList = os.listdir("./data/knn/trainingDigits")
    m = len(trainFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwlables.append(classNumStr)
        trainingMat[i] = img2vector("./data/knn/trainingDigits/%s" % fileNameStr)

    testFileList = os.listdir("./data/knn/testDigits")
    testLen = len(testFileList)
    errorCount = 0
    for i in range(testLen):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        testVect = img2vector("./data/knn/testDigits/%s" % fileNameStr)
        classifierResult = classify0(testVect,trainingMat,hwlables,3)
        print("the classifier came back with: %s, the real answer is: %d" % (str(classifierResult), classNumStr))
        errorCount += classifierResult != classNumStr
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / testLen))



if __name__ == '__main__':
    # datingClassTest()
    handwritingClassTest()