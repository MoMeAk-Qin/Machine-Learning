<br></br><br></br><br></br><br></br><br></br><center style="font-size:30px">哈尔滨工业大学计算机科学与技术学院 </center><center style="font-size:40px">实验报告 </center><br></br><br></br><br></br><br></br><center style="font-size:25px">课程名称： 机器学习</center><center style="font-size:25px">课程类型： 选修</center><center style="font-size:25px">实验题目： 逻辑回归</center><br></br><br></br><br></br><br></br><center style="font-size:20px">学号：</center><center style="font-size:20px">姓名：</center><div STYLE="page-break-after: always;"></div>

#### 一、实验目的

​         理解逻辑回归模型，掌握逻辑回归模型的参数估计算法。  

#### 二、实验要求及实验环境

###### 实验要求：

1. 实现两种损失函数的参数估计（1，无惩罚项；2.加入对参数的惩罚），可以采用梯度下降、共轭梯度或者牛顿法等。
2. 验证：可以手工生成两个分别类别数据（可以用高斯分布），验证你的算法。考察类条件分布不满足朴素贝叶斯假设，会得到什么样的结果。  
3. 验证：逻辑回归有广泛的用处，例如广告预测。可以到UCI网站上，找一实际数据加以测试。  

###### 实验环境：

​        Windows 10; Python3.7

#### 三、设计思想（本程序中的用到的主要算法及数据结构）

##### 1. 算法原理

本实验研究二分类问题：$f:X \rightarrow Y$，其中$X$为实数特征向量$<	X_{}1,...,X_{n}>$，$Y$为布尔型，即$Y\in{0,1}$，假设对于任意给定$Y$所有$X_{i}$条件独立，且$P(X_{i}|Y=y_{k})$服从高斯分布$N(\mu_{ik}, \sigma_{i})$，$P(Y)$服从伯努利分布$Bernoulli(\pi)$

Logistic Regression的==基本思想==是利用朴素贝叶斯分类器通过$P(Y)$和$P(X|Y)$来计算$P(Y|X)$。

可以推导出
$$
\begin{align}
P(Y=1|X)&=\frac{P(Y=1) P(X | Y=1)}{P(Y=1) P(X | Y=1)+P(Y=0) P(X | Y=0)}\\
&=\frac{1}{1+\frac{P(Y=0) P(X | Y=0)}{P(Y=1) P(X | Y=1)}}\\
&=\frac{1}{1+\exp \left(\ln \frac{P(Y=0)P(X | Y=0)}{P(Y=1)P(X | Y=1)}\right)}\\
\end{align}\tag{1}
$$
由于满足朴素贝叶斯假设且$P(Y)$服从伯努利分布可得$\hat{P}(Y=1)=\pi$带入式$(1)$得：
$$
P(Y=1|X)=\frac{1}{1+exp(\ln\frac{1-\pi}{\pi}+\sum_{i}\ln\frac{P(X_{i}|Y=0)}{P(X_{i}|Y=1)})}\tag{2}
$$
又由于$P(X_{i}|Y=y_{k})$服从高斯分布$N(\mu_{ik}, \sigma_{i})$即$P\left(x | y_{k}\right)=\frac{1}{\sigma_{i k} \sqrt{2 \pi}} e^{\frac{-\left(x-\mu_{i k}\right)^{2}}{2 \sigma_{i k}^{2}}}$带入式$(2)$得：
$$
P(Y=1|X)=\frac{1}{1+exp\{\ln\frac{1-\pi}{\pi}+\sum_{i}\left(\frac{\mu_{i 0}-\mu_{i 1}}{\sigma_{i}^{2}} X_{i}+\frac{\left.\mu_{i 1}^{2}-\mu_{i 0}^{2}\right)}{2 \sigma_{i}^{2}}\right)\}}\tag{3}
$$
可转化为：
$$
P(Y=1 | X)=\frac{1}{1+\exp \left(w_{0}+\sum_{i=1}^{n} w_{i} X_{i}\right)}\tag{4}
$$
由该问题是二分类问题可得：
$$
\begin{align}
P(Y=0 | X=<X_{1}, ...,X_{n}>)&=\frac{\exp \left(w_{0}+\sum_{i=1}^{n} w_{i} X_{i}\right)}{1+\exp \left(w_{0}+\sum_{i=1}^{n} w_{i} X_{i}\right)}\\
P(Y=1 | X=<X_{1}, ...,X_{n}>)&=\frac{1}{1+\exp \left(w_{0}+\sum_{i=1}^{n} w_{i} X_{i}\right)}
\end{align}\tag{5}
$$
即可得到一个$log-linear\ model$：$0 \lessgtr \ln \frac{P(Y=0 | X)}{P(Y=1 | X)}=w_{0}+\sum_{i} w_{i} X_{i}$。

为实现logistic regression根据Andrew Ng的公开课中所说需找到一个合适的预测函数(hypothesis)，该函数为我们所找的分类函数，根据上面的推到我们选择利用logistic函数(或称为sigmoid函数)，该函数将实数域的量映射到$(0,1)$区间，在上述二分类问题中有着良好的效果，其表达式为：$g(z)=\frac{1}{1+e^{-z}}$，该函数复合$X_{i}$的加权函数后，起到归一化作用可得到预测函数$h_{w}(x)=g(w^{T}x)=\frac{1}{1+e^{-w^{T} x}}$。

根据训练集$\left\{\left\langle X^{1}, Y^{1}\right\rangle, \ldots\left\langle X^{L}, Y^{L}\right\rangle\right\}$对参数$W$的最大似然估计$W_{MLE}$为：
$$
\begin{align}
W_{M L E}&=\arg \max _{W} P\left(<X^{1}, Y^{1}>\ldots<X^{L}, Y^{L}>| W\right)\\
&=\arg \max _{W} \prod_{l} P\left(<X^{l}, Y^{l}>| W\right)\\
\end{align}\tag{6}\\
$$
但是$\log L(D;W)=\log \prod_{i=1}L(X_{i},Y_{i};W)$即$arg \max _{W} \prod_{l} P\left(<X^{l}, Y^{l}>| W\right)$不易于求取，因此用最大条件似然求参数$W$即：
$$
W_{M C L E}=\arg \max _{W} \prod_{l} P\left(Y^{l} | X^{l},W\right)\tag{7}
$$
取条件似然函数为：
$$
L(W)=\prod_{l} P\left(Y^{l} | X^{l},W\right)\tag{8}
$$
则对数条件似然函数为：
$$
l(W) \equiv \ln \prod P\left(Y^{l} | X^{l}, W\right)=\sum \ln P\left(Y^{l} | X^{l}, W\right)\tag{9}
$$
其中：
$$
\begin{array}{l}{P(Y=0 | X, W)=\frac{1}{1+\exp \left(w_{0}+\sum_{i} w_{i} X_{i}\right)}} \\ {P(Y=1 | X, W)=\frac{\exp \left(w_{0}+\sum_{i} w_{i} X_{i}\right)}{1+\exp \left(w_{0}+\sum_{i} w_{i} X_{i}\right)}}\end{array}\tag{10}
$$
带入可得：
$$
\begin{aligned} l(W) &=\sum_{l} Y^{l} \ln P\left(Y^{l}=1 | X^{l}, W\right)+\left(1-Y^{l}\right) \ln P\left(Y^{l}=0 | X^{l}, W\right) \\ &=\sum_{l} Y^{l} \ln \frac{P\left(Y^{l}=1 | X^{l}, W\right)}{P\left(Y^{l}, W\right)}+\ln P\left(Y^{l}=0 | X^{l}, W\right) \\ &=\sum_{l} Y^{l}\left(w_{0}+\sum_{i}^{n} w_{i} X_{i}^{l}\right)-\ln \left(1+\exp \left(w_{0}+\sum_{i}^{n} w_{i} X_{i}^{l}\right)\right) \end{aligned}\tag{11}
$$
##### 2. 算法的实现

###### 梯度下降法

对数条件似然函数为：
$$
l(W)=\log L(W)=\sum_{i=1}^{m}\left(y^{(i)} \log h_{w}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{w}\left(x^{(i)}\right)\right)\right)\tag{12}
$$
损失函数$J(W)=-\frac{1}{m}l(W)$即：
$$
J(W)=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{w}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{w}\left(x^{(i)}\right)\right)\right]\tag{13}
$$
可以加入惩罚项即：
$$
J(W)=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{w}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{w}\left(x^{(i)}\right)\right)\right]+\frac{\lambda}{2m}W^{T}W \tag{13'}
$$


根据梯度下降法可得到$W$的更新过程：$W_{i}:=W_{i}-\alpha \frac{\partial}{\partial \theta_{i}} J(W), \quad(j=0 \ldots n)$其中$\alpha$为步长(学习率)。
$$
\begin{align}f(x)&=\frac{1}{1+e^{g(x)}}\\
\frac{\partial}{\partial x} f(x)&=\frac{1}{\left(1+e^{g(x)}\right)^{2}} e^{g(x)} \frac{\partial}{\partial x} g(x)\\
&=\frac{1}{1+e^{g(x)}} \frac{e^{g(x)}}{1+e^{g(x)}} \frac{\partial}{\partial x} g(x)\\
&=f(x)(1-f(x)) \frac{\partial}{\partial x} g(x)
\end{align}\tag{14}
$$
将式$(14)$带入梯度求解中，可以计算：
$$
\begin{align}
\frac{\partial}{\partial \theta_{j}} J(W)&=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)} \frac{1}{h_{w}\left(x^{(i)}\right)} \frac{\partial}{\partial w_{j}} h_{w}\left(x^{(i)}\right)-\left(1-y^{(i)}\right) \frac{1}{1-h_{w}\left(x^{(i)}\right)} \frac{\partial}{\partial w_{j}} h_{w}\left(x^{(i)}\right)\right)\\
&=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)} \frac{1}{g\left(w^{T} x^{(1)}\right)}-\left(1-y^{(i)}\right) \frac{1}{1-g\left(W^{T} x^{(i)}\right)}\right) \frac{\partial}{\partial w_{j}} g\left(w^{T} x^{(i)}\right)\\
&=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)} \frac{1}{g\left(w^{T} x^{(1)}\right)}-\left(1-y^{(i)}\right) \frac{1}{1-g\left(w^{T} x^{(i)}\right)}\right) g\left(w^{T} x^{(i)}\right)\left(1-g\left(w^{T} x^{(i)}\right)\right) \frac{\partial}{\partial \theta_{j}} w^{T} x^{(i)}\\
&=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)}\left(1-g\left(w^{T} x^{(i)}\right)\right)-\left(1-y^{(i)}\right) g\left(w^{T} x^{(i)}\right)\right) x_{j}^{(i)}\\
&=\frac{1}{m} \sum_{i=1}^{m}\left(h_{w}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
\end{align}\tag{15}
$$
因此更新过程可以写成：
$$
W_{j}:=W_{j}-\alpha \sum_{i=1}^{m}\left(h_{W}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}, \quad(j=0 \ldots n)\tag{16}
$$
将其向量化：
$$
x=\left[\begin{array}{c}{x^{(1)}} \\ {x^{(2)}} \\ {\cdots} \\ {x^{(\mathrm{m})}}\end{array}\right]=\left[\begin{array}{cccc}{x_{0}^{(1)}} & {x_{1}^{(1)}} & {\ldots} & {x_{n}^{(1)}} \\ {x_{0}^{(2)}} & {x_{1}^{(2)}} & {\ldots} & {x_{n}^{(2)}} \\ {\cdots} & {\cdots} & {\cdots} & {\cdots} \\ {x_{0}^{(\mathrm{m})}} & {x_{1}^{(\mathrm{m})}} & {\ldots} & {x_{n}^{(\mathrm{m})}}\end{array}\right] \quad, \quad y=\left[\begin{array}{c}{y^{(1)}} \\ {y^{(2)}} \\ {\cdots} \\ {y^{(\mathrm{m})}}\end{array}\right]\tag{17}
$$
带求参数$W$的矩阵为：
$$
W=\left[\begin{array}{c}{W_{0}} \\ {W_{1}} \\ {\cdots} \\ {W_{n}}\end{array}\right]\tag{18}
$$
则：
$$
A=x \cdot W=\left[\begin{array}{cccc}{x_{0}^{(1)}} & {x_{1}^{(1)}} & {\dots} & {x_{n}^{(1)}} \\ {x_{0}^{(2)}} & {x_{1}^{(2)}} & {\dots} & {x_{n}^{(2)}} \\ {\cdots} & {} & {\cdots} & {x_{n}^{(2)}} \\ {\cdots} & {\cdots} & {\cdots} & {\cdots} \\ {x_{0}^{(\mathrm{m})}} & {x_{1}^{(\mathrm{m})}} & {\dots} & {x_{n}^{(\mathrm{m})}}\end{array}\right] \cdot\left[\begin{array}{c}{W_{0}} \\ {W_{1}} \\ {\cdots} \\ {\theta_{n}}\end{array}\right]=\left[\begin{array}{c}{W_{0} x_{0}^{(1)}+W_{1} x_{1}^{(1)}+\ldots+W_{n} x_{n}^{(1)}} \\ {W_{0} x_{0}^{(2)}+W_{1} x_{1}^{(2)}+\ldots+W_{n} x_{n}^{(2)}} \\ {\cdots} \\ {\cdots} \\ {W_{0} x_{0}^{(\mathrm{m})}+W_{1} x_{1}^{(\mathrm{m})}+\ldots+W_{n} x_{n}^{(\mathrm{m})}}\end{array}\right]\\
E=h_{w}(\mathrm{x})-y=\left[\begin{array}{c}{g\left(A^{(1)}\right)-y^{(1)}} \\ {g\left(A^{(2)}\right)-y^{(2)}} \\ {g\left(A^{(\mathrm{m})}\right)-y^{(\mathrm{m})}}\end{array}\right]=\left[\begin{array}{c}{e^{(1)}} \\ {e^{(2)}} \\ {\cdots} \\ {e^{(\mathrm{m})}}\end{array}\right]=g(A)-y\\\tag{19}
$$
带入式$(16)$得：
$$
\begin{align}\left[\begin{array}{c}{W_{0}} \\ {W_{1}} \\ {\cdots} \\ {W_{n}}\end{array}\right]:&=\left[\begin{array}{c}{W_{0}} \\ {W_{1}} \\ {\cdots} \\ {W_{n}}\end{array}\right]-\alpha \cdot\left[\begin{array}{c}{x_{0}^{(1)}, x_{0}^{(2)}, \ldots, x_{0}^{(\mathrm{m})}} \\ {x_{1}^{(1)}, x_{1}^{(2)}, \ldots, x_{1}^{(\mathrm{m})}} \\ \cdots \\ {x_{n}^{(1)}, x_{n}^{(2)}, \ldots, x_{n}^{(\mathrm{m})}}\end{array}\right] \cdot \left[\begin{array}{c}{e^{(1)}} \\ {e^{(2)}} \\ {\cdots} \\ {e^{(\mathrm{m})}}\end{array}\right]\\
&=W-\alpha \cdot X^{T} \cdot (Sigmoid(X\cdot W)-Y)
\end{align}\tag{20}
$$
即可进行梯度下降发的求解。

###### 牛顿法

牛顿法的基本思想是利用迭代点出的一阶导数(梯度)和二阶导数($Hessian$矩阵)对目标函数进行二次近似，是一个二阶方法因此相较于梯度下降法来说速度更快。

其更新过程可以写为：
$$
W := W-H^{-1} \nabla\tag{21}
$$
梯度即为对数条件似然函数的梯度，在梯度下降法中已经导出，$Hessian$阵的求解为：
$$
\begin{aligned} H_{i j}=& \frac{\partial^{2} l(W)}{\partial W_{i} \partial W_{j}} \\ &=\frac{1}{m} \frac{\partial}{W_{j}} \sum_{t=1}^{m}\left(y^{(t)}-h_{w}\left(x^{(t)}\right)\right) x_{i}^{(t)} \\ &=\frac{1}{m} \sum_{t=1}^{m} \frac{\partial}{W_{j}}\left(y^{(t)}-h_{w}\left(x^{(t)}\right)\right) x_{i}^{(t)} \\ &=\frac{1}{m} \sum_{t=1}^{m}-x_{i}^{(t)} h_{w}\left(x^{(i)}\right)\left(1-h_{w}\left(x^{(i)}\right)\right) \frac{\partial}{W_{j}}\left(W^{T} x^{(t)}\right) \\ &=\frac{1}{m} \sum_{t=1}^{m} h_{w}\left(x^{(t)}\right)\left(h_{w}\left(x^{(t)}\right)-1\right) x_{i}^{(t)} x_{j}^{(t)} \end{aligned}\tag{22}
$$
可写为：
$$
Hessian Matrix = X\cdot diagonal(Sigmoid(XW)\cdot (Sigmoid(XW)-1))\cdot X^{T}\tag{23}
$$

#### 四、实验结果与分析

##### 1.满足朴素贝叶斯假设的两个独立且同分布的高斯分布数据结果

###### 不带正则项

梯度下降法

![Figure_1](D:\Document\CS doc\machine learning\lab2\Figure_1.png)
测试集分类效果检测正确率为$96.8\%$

牛顿法

![Figure_2](D:\Document\CS doc\machine learning\lab2\Figure_2.png)
测试集分类效果检测正确率为$97.6 \%$

###### 带正则项

梯度下降法

![Figure_3](D:\Document\CS doc\machine learning\lab2\Figure_3.png)

超参数$\lambda = 0.01$测试集分类效果检测正确率为$97.4\%$

牛顿法

![Figure_4](D:\Document\CS doc\machine learning\lab2\Figure_4.png)
超参数$\lambda = 0.01$测试集分类效果检测正确率为$98.3\%$

##### 2.不满足朴素贝叶斯假设的两个独立且同分布的高斯分布数据结果

###### 不带正则项

梯度下降法

![Figure_5](D:\Document\CS doc\machine learning\lab2\Figure_5.png)
测试集分类效果检测正确率为$96.4 \%$

牛顿法

![Figure_6](D:\Document\CS doc\machine learning\lab2\Figure_6.png)
测试集分类效果检测正确率为$96.5 \%$

###### 带正则项

梯度下降法

![Figure_7](D:\Document\CS doc\machine learning\lab2\Figure_7.png)
超参数$\lambda = 0.01$测试集分类效果检测正确率为$96.4\%$

牛顿法

![Figure_8](D:\Document\CS doc\machine learning\lab2\Figure_8.png)

超参数$\lambda = 0.01$测试集分类效果检测正确率为$96.6\%$

由以上结果可发现无论数据是否满足朴素贝叶斯假设，分类效果都均在$95\%$以上，效果较好，且梯度下降法和牛顿法二者的效果无论在带正则项和不带正则项的情况下，效果十分接近。 

##### 3.使用uci数据(天平平衡数据集)

数据样式为：

![1571684841602](C:\Users\57363\AppData\Roaming\Typora\typora-user-images\1571684841602.png)
其中量意义为：

```
1. Class Name: 3 (L, B, R)
2. Left-Weight: 5 (1, 2, 3, 4, 5)
3. Left-Distance: 5 (1, 2, 3, 4, 5)
4. Right-Weight: 5 (1, 2, 3, 4, 5)
5. Right-Distance: 5 (1, 2, 3, 4, 5)
```
只选取左倾或者右倾的数据形成二分类问题，选取400个数据作为训练集，100个数据作为测试集。

使用**梯度下降法**的分类效果为$77\%$。

造成这一分类结果是由于数据量较小且自然噪声较大(数据是有一定规律排序的，前后分割训练集和测试集科学，应随机选取，但是随机选取导致多次实验时，每次的训练集和测试集又都不相同很难对照)。

#### 五、结论

1. 梯度下降法和牛顿法在实现logistic回归时均可以实现较好的分类结果；
2. 加入惩罚项后并不能带来更大的提升(较线性拟合而言)；
3. 牛顿法作为二阶方法收敛的速度比梯度下降法要快得多；
4. 当数据不满足朴素贝叶斯假设时，分类效果依旧很好；
5.  对于某些类型的概率模型，在监督式学习的样本集中能获取得非常好的分类效果。在许多实际应用中，朴素贝叶斯模型参数估计使用最大似然估计方法；换而言之，在不用到贝叶斯概率或者任何贝叶斯模型的情况下，朴素贝叶斯模型也能奏效。  朴素贝叶斯分类器的一个优势在于只需要根据少量的训练数据估计出必要的参数（变量的均值和方差）。由于变量独立假设，只需要估计各个变量的方法，而不需要确定整个协方差矩阵；
6. 朴素贝叶斯假设，也就是条件独立，即在给定样本的下几个维度互相独立。在朴素贝叶斯假设下，边缘分布与单独变量的分布相同，打破这一假设后依旧线性可分，但是条件独立性的打破使得我们不能只关注分布来用边缘获联合，而是要关注整个联合分布。

#### 六、参考文献

1. 《机器学习》 周志华.北京.清华大学出版社
2.  https://datawhalechina.github.io/leeml-notes/#/chapter11/chapter11 
3.  https://blog.csdn.net/achuo/article/details/51160101 

#### 七、附录：源代码（带注释）

lab2.py

**该文件为手工生成数据的判别**

```python
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(inx):
    return 1 / (1 + np.exp(-inx))


def model(X, W):
    '''
    预测函数
    '''
    return sigmoid(np.dot(X, W))


def cost_function(W, X, Y):
    '''
    cost1为对数似然函数
    cost2为损失函数
    '''
    cost = -np.dot(Y.T, np.log(model(X, W))) - np.dot(
        (np.ones((scale_of_example * 2, 1)) - Y).T,
        np.log(np.ones((scale_of_example * 2, 1)) - model(X, W)),
    )
    cost = sum(cost) / len(X) + (hyper_parameter * (np.dot(W.T, W))) / (2 * len(X))
    print(cost)
    return cost


def gradient_function(W, X, Y):
    '''
    计算梯度
    '''
    gradient = np.dot(X.T, (model(X, W) - Y))
    return gradient


def gradient_decent(W, example_added, label, threshold):
    '''
    梯度下降法
    '''
    alpha = 0.0001
    gradient = gradient_function(W, example_added, label)
    while cost_function(W, example_added, label) > threshold:
        W = W - alpha * gradient
        gradient = gradient_function(W, example_added, label)
    return W


def Hessian(W, X, Y):
    '''
    生成黑塞矩阵
    '''
    hessianMatrix = np.zeros((dim + 1, dim + 1))
    for t in range(scale_of_example * 2):
        X_mat = np.mat(X[t]).T
        XXT = np.array(X_mat * X_mat.T)
        hessianMatrix += sigmoid(np.dot(X[t], W)) * (sigmoid(np.dot(X[t], W)) - 1) * XXT
    return hessianMatrix


def newton_method(W, example_added, label, threshold):
    '''
    牛顿法
    '''
    gradient = gradient_function(W, example_added, label)
    alpha = 0.01
    while cost_function(W, example_added, label) > threshold:
        H = np.linalg.inv(Hessian(W, example_added, label))
        W = W + alpha * np.dot(H, gradient)
        gradient = gradient_function(W, example_added, label)
    return W


def judge(W):
    '''
    判断回归效果 
    '''
    judge_scale = 500
    s1 = np.dot(np.random.randn(judge_scale, dim), R1) + mu1
    plt.plot(s1[:, 0], s1[:, 1], "+", label="test_set1", color="b")
    s2 = np.dot(np.random.randn(judge_scale, dim), R2) + mu2
    plt.plot(s2[:, 0], s2[:, 1], "+", label="test_set2", color="g")
    example = np.vstack((s1, s2))
    label1 = np.zeros((judge_scale, 1))
    label2 = np.ones((judge_scale, 1))
    test_label = np.vstack((label1, label2))
    test_set = np.hstack((np.ones((judge_scale * 2, 1)), example))
    result = np.zeros((judge_scale * 2, 1))
    correct_num = 0
    for i in range(judge_scale * 2):
        if model(test_set, W)[i - 1][0] > 0.5:
            result[i - 1][0] = 1
    for i in range(judge_scale * 2):
        if result[i - 1][0] == test_label[i - 1][0]:
            correct_num += 1
    return correct_num / (judge_scale * 2)


if __name__ == "__main__":
    '''
    生成训练数据，为二维随机高斯分布
    label为二分类分别为0和1
    hyper_parameter为\lambda
    '''
    scale_of_example = 1000
    dim = 2
    mu1 = np.array([[1, 3]])
    Sigma1 = np.array([[1, 0], [0, 3]])
    R1 = np.linalg.cholesky(Sigma1)
    s1 = np.dot(np.random.randn(scale_of_example, dim), R1) + mu1
    # plt.plot(s1[:, 0], s1[:, 1], ".", label="training_set1", color="red")

    mu2 = np.array([[4, 7]])
    Sigma2 = np.array([[1, 0], [0, 3]])
    R2 = np.linalg.cholesky(Sigma2)
    s2 = np.dot(np.random.randn(scale_of_example, dim), R2) + mu2
    # plt.plot(s2[:, 0], s2[:, 1], ".", label="training_set2", color="yellow")

    example = np.vstack((s1, s2))
    label1 = np.zeros((scale_of_example, 1))
    label2 = np.ones((scale_of_example, 1))
    label = np.vstack((label1, label2))
    data = np.hstack((example, label))
    W = np.ones((dim + 1, 1))

    hyper_parameter = 0.0001

    example_added = np.hstack((np.ones((scale_of_example * 2, 1)), example))
    cost_function(W, example_added, label)

    # W = gradient_decent(W, example_added, label, 0.1)     # 梯度下降法
    W = newton_method(W, example_added, label, 0.1)     # 牛顿法
    print(judge(W))
    X1 = np.linspace(-2, 10, 20)
    X2 = -W[0][0] / W[2][0] - np.dot(W[1][0], X1) / W[2][0]
    plt.plot(X1, X2, label="gd_regularized", color="m")
    plt.legend()
    plt.show()
```

uci.py

**该文件为使用uci网站数据的判别**

```python
import numpy as np
import io
import re
import operator


def sigmoid(inx):
    return 1 / (1 + np.exp(-inx))


def model(X, W):
    return sigmoid(np.dot(X, W))


def cost_function(W, X, Y):
    cost = -np.dot(Y.T, np.log(model(X, W))) - np.dot(
        (np.ones((scale_of_example * 2, 1)) - Y).T,
        np.log(np.ones((scale_of_example * 2, 1)) - model(X, W)),
    )
    cost = sum(cost) / len(X) + (hyper_parameter * (np.dot(W.T, W))) / (2 * len(X))
    print(cost)
    return cost


def gradient_function(W, X, Y):
    gradient = np.dot(X.T, (model(X, W) - Y))
    return gradient


def gradient_decent(W, example_added, label, threshold):
    alpha = 0.0001
    gradient = gradient_function(W, example_added, label)
    while cost_function(W, example_added, label) > threshold:
        W = W - alpha * gradient
        gradient = gradient_function(W, example_added, label)
    return W


def judge(W):
    judge_scale = 50
    result = np.zeros((judge_scale * 2, 1))
    correct_num = 0
    for i in range(judge_scale * 2):
        if model(test_set, W)[i - 1][0] > 0.5:
            result[i - 1][0] = 1
    for i in range(judge_scale * 2):
        if result[i - 1][0] == test_label[i - 1][0]:
            correct_num += 1
    return correct_num / (judge_scale * 2)


if __name__ == "__main__":
    bl = io.open("balance-scale.data", encoding="UTF-8")
    bl_list = bl.readlines()
    scale_of_example = 200
    dim = 4
    example = [1, 1, 1, 1, 1]
    label = []
    i = 0
    '''
    二分类只选取左倾或者右倾的数据
    平衡状态不考虑
    选取前400个数据训练
    后一百个数据测试
    '''
    for line in bl_list:
        l = re.split('[,\n]', line)
        if operator.eq(l[0], 'L'):
            label.append(0)
            tmp = np.mat([1, int(l[1]), int(l[2]), int(l[3]), int(l[4])])
            example = np.vstack((example, tmp))
        elif operator.eq(l[0], 'R'):
            label.append(1)
            tmp = np.mat([1, int(l[1]), int(l[2]), int(l[3]), int(l[4])])
            example = np.vstack((example, tmp))
        else:
            continue
        i = i + 1

    example_added = example[1:401, :]
    test_set = example[401:501, :]
    training_label = np.mat(label).T[1:401, :]
    test_label = np.mat(label).T[401:501, :]
    label = training_label
    W = np.zeros((dim + 1, 1))
    hyper_parameter = 0.00001
    cost_function(W, example_added, label)
    W = gradient_decent(W, example_added, label, 0.402)
    print(judge(W))
```

