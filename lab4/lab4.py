# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky
from mpl_toolkits.mplot3d import Axes3D


def pca(datamat, top):
    # 对所有样本进行中心化（所有样本属性减去属性的平均值）
    meanVals = np.mean(datamat, axis=0)
    meanRemoved = datamat - meanVals
    # 计算样本的协方差矩阵 XXT
    covmat = np.cov(meanRemoved, rowvar=0)
    # 对协方差矩阵做特征值分解，求得其特征值和特征向量，
    # 并将特征值从大到小排序，筛选出前top个
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    print(eigVals)
    eigValInd = np.argsort(eigVals)
    # 取前top大的特征值的索引
    eigValInd = eigValInd[: -(top + 1) : -1]
    # 取前top大的特征值所对应的特征向量
    redEigVects = eigVects[:, eigValInd]
    # 将数据转换到新的低维空间中
    # 降维之后的数据
    lowDDataMat = meanRemoved * redEigVects
    # 重构数据，可在原数据维度下进行对比查看
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return reconMat, lowDDataMat, redEigVects


sampleNo = 500
dim = 3
mu1 = np.array([[1, 5, 3]])
Sigma1 = np.array([[50, 19, 2], [19, 40, 9], [2, 9, 15]])
R1 = cholesky(Sigma1)
datamat = np.dot(np.random.randn(sampleNo, dim), R1) + mu1
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(datamat[:, 0], datamat[:, 1], datamat[:, 2], alpha=0.4, s=10)
plt.ylim((-10, 15))
plt.xlim(-10, 15)
ax.set_zlim(-10, 15)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
reconMat, lowDDataMat, red = pca(datamat, 2)
# ax.scatter(lowDDataMat[:, 0], lowDDataMat[:, 1], alpha=0.4, s=10)
ax.scatter(reconMat[:, 0], reconMat[:, 1], reconMat[:, 2], alpha=0.4, s=10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("performance on 3-demensional space")
plt.show()
