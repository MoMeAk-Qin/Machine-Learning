import matplotlib.pyplot as plt
import random
import numpy as np

"""
生成三组二维的高斯分布
"""
scale_of_example = 100
dimension = 2
mu1 = np.array([[0, 4]])
sigma1 = np.array([[2, 0], [0, 2]])
R1 = np.linalg.cholesky(sigma1)
x1 = np.dot(np.random.randn(scale_of_example, dimension), R1) + mu1
mu2 = np.array([[4, 6]])
sigma2 = np.array([[2, 0], [0, 2]])
R2 = np.linalg.cholesky(sigma2)
x2 = np.dot(np.random.randn(scale_of_example, dimension), R2) + mu2
mu3 = np.array([[2, -2]])
sigma3 = np.array([[2, 0], [0, 2]])
R3 = np.linalg.cholesky(sigma3)
x3 = np.dot(np.random.randn(scale_of_example, dimension), R3) + mu3

"""
合并三组数据
"""
temp1 = np.vstack((x1, x2))
data = np.vstack((temp1, x3))
"""
求两点的欧氏距离
"""
def eudlidean_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


"""
随机初始质心
"""
random1 = random.randint(0, 299)
random2 = random.randint(0, 299)
random3 = random.randint(0, 299)
center1 = data[random1, :]
center2 = data[random2, :]
center3 = data[random3, :]
"""
设置标签位(标签位用于最后画图时区分不同类别)
"""
label_c = np.zeros((300, 1))
"""
k-means
"""
while True:
    sum1 = np.zeros((1, 2))
    sum2 = np.zeros((1, 2))
    sum3 = np.zeros((1, 2))
    num1 = 0
    num2 = 0
    num3 = 0
    for i in range(300): # 分类
        if eudlidean_distance(center1, data[i, :]) < eudlidean_distance(
            center2, data[i, :]
        ) and eudlidean_distance(center1, data[i, :]) < eudlidean_distance(center3, data[i, :]):
            label_c[i, 0] = 1
            sum1 = sum1 + data[i, :]
            num1 = num1 + 1
        elif eudlidean_distance(center3, data[i, :]) < eudlidean_distance(
            center2, data[i, :]
        ) and eudlidean_distance(center3, data[i, :]) < eudlidean_distance(center1, data[i, :]):
            label_c[i, 0] = 0
            sum3 = sum3 + data[i, :]
            num3 = num3 + 1
        else:
            label_c[i, 0] = 2
            sum2 = sum2 + data[i, :]
            num2 = num2 + 1
    sum1 = sum1 / num1
    sum2 = sum2 / num2
    sum3 = sum3 / num3
    threshold = 1e-15
    if (    # 当质心不再改变时迭代结束
        eudlidean_distance(sum3, center3) <= threshold
        and eudlidean_distance(sum1, center1) <= threshold
        and eudlidean_distance(sum2, center2) <= threshold
    ):
        break
    """
    更新质心
    """
    center1 = sum1
    center2 = sum2
    center3 = sum3
for i in range(300):
    if label_c[i, 0] == 1:
        sum1 = np.vstack((sum1, data[i, :]))
    elif label_c[i, 0] == 2:
        sum2 = np.vstack((sum2, data[i, :]))
    else:
        sum3 = np.vstack((sum3, data[i, :]))
"""
画图输出
"""
plt.plot(sum1[:, 0], sum1[:, 1], ".", label="training_set1", color="b")
plt.plot(sum2[:, 0], sum2[:, 1], ".", label="training_set2", color="g")
plt.plot(sum3[:, 0], sum3[:, 1], ".", label="training_set3", color="r")

plt.plot(center1[:, 0], center1[:, 1], "*", label="training_set1", color="y")
plt.plot(center2[:, 0], center2[:, 1], "+", label="training_set1", color="y")
plt.plot(center3[:, 0], center3[:, 1], "s", label="training_set1", color="y")
plt.show()
