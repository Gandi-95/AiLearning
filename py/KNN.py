import numpy as np
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

def main():
    returnMat, classLabelVector =file2Matrix("D:\PythonProjects\AiLearning\data\KNNData")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:,0],returnMat[:,1], 15.0* np.array(classLabelVector), 15.0*np.array(classLabelVector))
    plt.show()



if __name__ == '__main__':
    main()