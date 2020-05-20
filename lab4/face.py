# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


dataset = []
for i in range(1, 41):
    for j in range(1, 11):
        img = cv2.imread(".//ORL//" + str(i) + "_" + str(j) + "_small.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((1, 2576))
        dataset.append(img)
dataset = np.reshape(np.array(dataset), [-1, 2576])


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


def PSNR(origin, pcaed, pix1, pix2):
    MSE = sum(np.dot(pcaed - origin, (pcaed - origin).T)) / (pix1 * pix2)
    ans = 10 * np.log10(255 * 255 / MSE)
    return ans


plt.figure()
train = 100
test = 200
reconMat, lowDDataMat, redEigVects, a = pca(dataset[0:train, :], 98)
set = dataset[train : train + test, :]
meanvals = np.mean(set, axis=0)
data_rem = set - meanvals
lowDDataMat = data_rem * redEigVects
reconMat = (lowDDataMat * redEigVects.T) + meanvals
#print(reconMat.shape)
pic = reconMat[1].reshape((56, 46))
print("PSNR = ")
print(abs(PSNR(set[1], reconMat[1], 56, 46)))
pic = pic.astype(np.uint8)
cv2.imshow("pic", pic)
cv2.waitKey()