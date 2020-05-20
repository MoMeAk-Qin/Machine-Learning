import matplotlib.pyplot as plt
import random
import numpy as np
from numpy.linalg import cholesky


def normal_sim(x, u, sigma):
    delta = np.linalg.det(sigma)
    ans = (
        np.exp(-np.dot(np.dot((x - u).T, np.linalg.inv(sigma).T), (x - u)) / 2) / delta
    )
    return ans


def judge(la):
    print(la)
    sum2 = np.sum(la == 3)
    sum0 = np.sum(la == 1)
    sum1 = np.sum(la == 2)
    print(sum1, sum0, sum2)
    if sum1 > sum2 and sum1 > sum0:
        count = sum2 + sum0
    elif sum0 > sum1 and sum0 > sum2:
        count = sum1 + sum2
    else:
        count = sum1 + sum0
    return count


def EM(data, k, scale, dimension, label):
    """
    初始化
    """
    alpha = np.ones((1, k)) / k
    rand = np.zeros((1, k))
    for i in range(k):
        rand[0, i] = random.randint(0, scale)
    mu = np.zeros((k, dimension))
    for i in range(k):
        a = int(rand[0, i])
        mu[i, :] = data[a, :]
    sig = np.zeros((dimension, dimension, k))
    sig_temp = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            if i == j:
                sig_temp[i, j] = 1
    for i in range(k):
        sig[:, :, i] = sig_temp
    temp = np.zeros((1, k))
    gama = np.zeros((scale, k))
    it = 70    # 最大迭代次数
    for t in range(it):
        change = 0
        # E-step
        for j in range(scale):
            sum = 0
            for i in range(k):
                temp[0, i] = alpha[0, i] * normal_sim(
                    np.mat(data[j, :]).T, np.mat(mu[i, :]).T, np.mat(sig[:, :, i])
                )
                sum = sum + temp[0, i]
            for i in range(k):
                gama[j, i] = temp[0, i] / sum
        # M-step
        for i in range(k):
            sum = 0
            sum_mu = np.zeros((1, dimension))
            sum_sigma = np.zeros((dimension, dimension))
            for j in range(scale):
                sum = sum + gama[j, i]
                sum_mu = sum_mu + gama[j, i] * data[j, :]
            change = change + float(
                np.dot(mu[i, :] - sum_mu / sum, (mu[i, :] - sum_mu / sum).T)
            )
            mu[i, :] = sum_mu / sum # 更新mu
            change = change + (np.sum((alpha[0, i] - sum / scale) ** 2))
            alpha[0, i] = sum / scale # 更新alpha
            for j in range(scale):
                deta = np.mat(data[j, :] - mu[i, :])
                sum_sigma = sum_sigma + gama[j, i] * np.dot(deta.T, deta)
            change = change + (np.sum((sig[:, :, i] - sum_sigma / sum) ** 2))
            sig[:, :, i] = sum_sigma / sum # 更新sigma
        if change < 1e-6: # 停止迭代
            print("The number of iteration until break:")
            print(t)
            break

    type0 = np.zeros((1000, dimension + 1))
    type1 = np.zeros((1000, dimension + 1))
    type2 = np.zeros((1000, dimension + 1))
    n0 = 0
    n1 = 0
    n2 = 0
    print(label)
    for j in range(scale):
        if gama[j, 0] > gama[j, 1] and gama[j, 0] > gama[j, 2]:
            type0[n0, 0:dimension] = data[j, :]
            type0[n0, dimension] = label[j, 0]
            n0 = n0 + 1
        elif gama[j, 1] > gama[j, 2] and gama[j, 1] > gama[j, 0]:
            type1[n1, 0:dimension] = data[j, :]
            type1[n1, dimension] = label[j, 0]
            n1 = n1 + 1
        else:
            type2[n2, 0:dimension] = data[j, :]
            type2[n2, dimension] = label[j, 0]
            n2 = n2 + 1
    error = (
        judge(type1[1:n1, dimension])
        + judge(type0[1:n0, dimension])
        + judge(type2[1:n2, dimension])
    )
    print(error)
    return 1 - error / scale, type0, n0, type1, n1, type2, n2, mu


if __name__ == "__main__":
    # scale_of_example = 100
    # dimensionension = 2
    # mu1 = np.array([[0, 4]])
    # sigma1 = np.array([[2, 0], [0, 2]])
    # R1 = cholesky(sigma1)
    # x1 = np.dot(np.random.randn(scale_of_example, dimensionension), R1) + mu1
    # mu2 = np.array([[4, 6]])
    # sigma2 = np.array([[2, 0], [0, 2]])
    # R2 = cholesky(sigma2)
    # x2 = np.dot(np.random.randn(scale_of_example, dimensionension), R2) + mu2
    # mu3 = np.array([[2, -2]])
    # sigma3 = np.array([[2, 0], [0, 2]])
    # R3 = cholesky(sigma3)
    # x3 = np.dot(np.random.randn(scale_of_example, dimensionension), R3) + mu3
    # cmp = np.vstack((mu1, mu2))
    # cmp = np.vstack((cmp, mu3))

    # data0 = np.vstack((x1, x2))
    # data = np.vstack((data0, x3))
    # label0 = np.vstack(
        # (
            # np.ones((scale_of_example, 1)),
            # np.ones((scale_of_example, 1)) + np.ones((scale_of_example, 1)),
        # )
    # )
    # label = np.vstack(
        # (
            # label0,
            # np.ones((scale_of_example, 1))
            # + np.ones((scale_of_example, 1))
            # + np.ones((scale_of_example, 1)),
        # )
    # )
    # print("The EM performance on the generated GMM")
    # accuracy, type0, n0, type1, n1, type2, n2, mu = EM(data, 3, 300, 2, label)
    # print("The accuracy :")
    # print(accuracy)
    # print("visualization")
    # plt.plot(type0[0:n0, 0], type0[0:n0, 1], ".", color="red")
    # plt.plot(type1[0:n1, 0], type1[0:n1, 1], ".", color="blue")
    # plt.plot(type2[0:n2, 0], type2[0:n2, 1], ".", color="green")
    # plt.scatter(mu[:, 0], mu[:, 1], color="black")
    # plt.show()
    # print()
    # print("The EM performance on the uci data set")
    uci_data = np.loadtxt("seeds_dataset.txt")
    uci_data = np.mat(uci_data)
    accuracy, type0, n0, type1, n1, type2, n2, mu = EM(uci_data[:, :7], 3, 210, 7, np.mat(uci_data[:, 7]))
    print("The accuracy:")
    print(accuracy)
