import numpy as np
import matplotlib.pyplot as plt
import random


def without_regular_term(x, Y):
    '''
    不带正则项的解析解
    W = (X.T * X).I * X.T * T
    '''
    A = x.T * x
    A = A.I
    w = A * x.T * Y
    return w


def with_regular_term(x, Y, degree):
    '''
    带正则项的解析解
    W = (X.T * X + lambda * E(m+1)).I * X.T * T
    '''
    A = x.T * x + 0.001 * np.eye(degree)
    A = A.I
    w = A * x.T * Y
    return w


def loss(W, X, Y, lamda):
    '''
    损失函数
    '''
    return (1 / X.shape[0]) * ((X * W - Y).T * (X * W - Y) + lamda * W.T * W)


def gradient(W, X, Y, lamda):
    '''
    计算带正则项的梯度
    '''
    return (1 / X.shape[0]) * (X.T * X * W - X.T * Y + lamda * W)


def gradient_descent(X, Y, learning_rate, lamda):
    time = 0
    W = np.mat(np.zeros(X.shape[0])).T
    grad = gradient(W, X, Y, lamda)
    while np.all(np.absolute(loss(W, X, Y, lamda)) > 1e-5):
        W = W - learning_rate * grad
        grad = gradient(W, X, Y, lamda)
        time += 1
        if time >= 100000:
            break
    return W


def conjugate_gradient(X, Y, W):
    '''
    共轭梯度法
    形如Ax=b，其中A是对称正定的
    '''
    A = X.T * X + 0.01 * np.eye(101)
    b = X.T * Y
    r = b - A * W
    p = r
    for i in range(1, 100):
        alpha = (r.T * r) / (p.T * A * p)
        r_prev = r
        W = W + np.multiply(alpha, p)
        r = r - np.multiply(alpha , A * p)
        if all(abs(r)) <= 0.01:
            break
        beta = (r.T * r) / (r_prev.T * r_prev)
        p = r + np.multiply(beta, p)
    return W


if __name__ == '__main__':
    X = np.arange(-1, 1.02, 0.02)   #生成x，y
    Y = np.sin(2 * np.pi * X)
    plt.plot(X, Y, color="black", linewidth=2, label='sin(2πx)')
    mu = 0
    sigma = 0.15
    scale = X.size
    degree = 9
    learning_rate = 0.01
    lamda = 0.001
    W = np.mat(np.zeros((101, 1)))

    for i in range(scale):
        Y[i] += random.gauss(mu, sigma)     #加高斯噪声
    plt.scatter(X, Y, marker='.', color='blue', s=20, label='data')

    poly = np.polyfit(X, Y, deg=9)      #numpy拟合函数，用作对比
    z = np.polyval(poly, X)
    plt.plot(X, z, color='purple', label='fit_sin(2πx)')

    X = np.mat(X).T
    y = np.mat(Y).T
    x = np.power(X, 0)
    x_0 = np.power(X, 0)

    test_set = np.linspace(-1, 1, 1000)

    for i in range(1, degree):
        temp = np.power(X, i)
        x_0 = np.hstack((x_0, temp))        #范德蒙德行列式

    cofficient_matrix = with_regular_term(x_0, y, degree)
    func = 0 * test_set
    for i in range(0, degree):
        func += (np.double(cofficient_matrix[i])) * (test_set ** (i))
    plt.plot(test_set, func, color="red", linewidth=2, label='with_ regular_term')  #带正则项的曲线

    cofficient_matrix = without_regular_term(x_0, y)
    func = 0 * test_set
    for i in range(0, degree):
        func += (np.double(cofficient_matrix[i])) * (test_set ** (i))
    plt.plot(test_set, func, color="orange", linewidth=2, label='without_ regular_term')    #不带正则项的曲线

    for i in range(1, x.shape[0]):
        temp = np.power(X, i)
        x = np.hstack((x, temp))

    W = gradient_descent(x, y, learning_rate, lamda)

    func = 0 * test_set
    for i in range(0, x.shape[0]):
        func += (np.double(W[i])) * (test_set**i)
    plt.plot(test_set, func, color="green", linewidth=2, label='gradient_descent')  #梯度下降法

    W = np.mat(np.zeros((101, 1)))
    W = conjugate_gradient(x, y, W)
    func = 0 * test_set
    for i in range(0, x.shape[0]):
        func += (np.double(W[i])) * (test_set ** i)
    plt.plot(test_set, func, color="yellow", linewidth=2, label='conjugate_gradient')    #共轭梯度法


    plt.xlabel("label")
    plt.ylabel("sample")
    plt.legend(loc='best')
    plt.show()
