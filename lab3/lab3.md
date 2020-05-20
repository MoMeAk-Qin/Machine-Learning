<br></br><br></br><br></br><br></br><br></br><center style="font-size:30px">哈尔滨工业大学计算机科学与技术学院 </center><center style="font-size:40px">实验报告 </center><br></br><br></br><br></br><br></br><center style="font-size:25px">课程名称： 机器学习</center><center style="font-size:25px">课程类型： 选修</center><center style="font-size:25px">实验题目： GMM模型</center><br></br><br></br><br></br><br></br><center style="font-size:20px">学号：</center><center style="font-size:20px">姓名：</center><div STYLE="page-break-after: always;"></div>

#### 一、实验目的
实现一个k-means算法和混合高斯模型，并且用EM算法估计模型中的参数。
#### 二、实验要求及实验环境

###### 实验要求：

测试：

​	用高斯分布产生k个高斯分布的数据（不同均值和方差）（其中参数自己设定）。

 1. 用k-means聚类，测试效果；

 2. 用混合高斯模型和你实现的EM算法估计参数，看看每次迭代后似然值变化情况，考察EM算法是否可以获得正确的结果（与你设定的结果比较）。

应用：可以UCI上找一个简单问题数据，用你实现的GMM进行聚类。  

###### 实验环境：

Windows 10; Python3.7

#### 三、设计思想（本程序中的用到的主要算法及数据结构）

##### 1. 算法原理

###### k-means

$k$均值聚类是基于样本集合划分的聚类算法，$k$均值聚类将样本集合划分为$k$个子集，构成$k$个类，将$n$个样本分到$k$个类中，每个样本到其所属类的中心的距离最小。

给定n个样本的集合$X=\{x_{1},x_{2},\dotsc,x_{n}\}$和分类数量$k$，给出一个划分$C$，一个划分对应着一个聚类结果。$k-means$的策略是通过损失函数的最小化选取最优的划分或函数$C^{*}$。

采用欧氏距离作为样本之间的距离$d(x_{i},x_{j})$
$$
\begin{aligned} d\left(x_{i}, x_{j}\right) &=\sum_{k=1}^{m}\left(x_{k i}-x_{k j}\right)^{2} \\ &=\left\|x_{i}-x_{j}\right\|^{2} \end{aligned}\tag{1}
$$
定义损失函数为样本与其所属类的中心之间距离的总和
$$
W(C)=\sum_{l=1}^{k} \sum_{C(i)=l}\left\|x_{i}-\bar{x}_{l}\right\|^{2}\tag{2}
$$
则$k-means$就是求解最优化问题
$$
\begin{aligned} C^{*} &=\arg \min _{C} W(C) \\ &=\arg \min _{C} \sum_{l=1}^{k} \sum_{C(i)=l}\left\|x_{i}-\bar{x}_{l}\right\|^{2} \end{aligned}\tag{3}
$$
事实上，$k-means$的最优解求解问题是$NP$困难问题，在实际解决中采用迭代方法求解。

具体过程为：

首先对于给定的中心$(m_{1},m_{2}, \dotsc ,m_{k})$求一个划分$C$，使得目标函数极小化
$$
\min _{C} \sum_{l=1}^{k} \sum_{C(i)=l}\left\|x_{i}-m_{l}\right\|^{2}\tag{4}
$$
然后对给定的划分$C$，再求各个类的中心$(m_{1},m_{2}, \dotsc ,m_{k})$，使得目标函数极小化
$$
\min _{m_{1}, \cdots, m_{k}} \sum_{l=1}^{k} \sum_{C(i)=l}\left\|x_{i}-m_{l}\right\|^{2}\tag{5}
$$
就是说在划分确定的情况下，使样本和其所属类的中心之间的距离总和最小。求解结果，对于每个包含$n_{l}$个样本的类$C_{l}$，更新其均值$m_{l}$：
$$
m_{l}=\frac{1}{n_{l}} \sum_{C(i)=l} x_{i}, \quad l=1, \cdots, k\tag{6}
$$
重复上述两个步骤，直到划分不再改变，得到聚类结果。

###### GMM中的EM算法

当模型含有隐变量时，就不能简单地使用极大似然估计法或贝叶斯估计法估计模型参数，EM算法就是含有隐变量的概率模型参数的极大似然估计法，或极大后验概率估计法。一般地，用$Y$表示观测随机变量的数据，$Z$表示隐随机变量的数据，$\theta$是需要估计的模型参数。

$EM$算法是通过迭代求$L(\theta)=logP(Y|\theta)$的极大似然估计。每次迭代都包含两步：

$E$步，记$\theta^{(i)}$为第$i$次迭代参数$\theta$的估计值，在第$i+1$次迭代的$E$步，计算
$$
\begin{aligned} Q\left(\theta, \theta^{(i)}\right) &=E_{Z}\left[\log P(Y, Z | \theta) | Y, \theta^{(i)}\right] \\ &=\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log P(Y, Z | \theta) \end{aligned}\tag{7}
$$
$M$步，求使$Q\left(\theta, \theta^{(i)}\right)$极大化的$\theta$，确定第$i+1$次迭代的参数估计值$\theta^{(i+1)}$
$$
\theta^{(i+1)}=\arg \max _{\theta} Q\left(\theta, \theta^{(i)}\right)\tag{8}
$$
那么可以通过近似求解观测数据的对数似然函数的极大化问题来导出$EM$算法。

对一个含有隐变量的概率模型，目标是极大化观测数据$Y$关于参数$\theta$的对数似然函数，即最大化：
$$
\begin{aligned} L(\theta) &=\log P(Y | \theta)=\log \sum_{Z} P(Y, Z | \theta) \\ &=\log \left(\sum_{Z} P(Y | Z, \theta) P(Z | \theta)\right) \end{aligned}\tag{9}
$$
事实上，$EM$算法是通过迭代逐步近似极大化$L(\theta)$的。假设在第$i$次迭代后$\theta$的估计值是$\theta^{(i)}$。希望估计值$\theta$能使$L(\theta)$增加，即$L(\theta)>L\left(\theta^{(i)}\right)$，并逐步达到极大值。为此考虑二者的差值：
$$
L(\theta)-L\left(\theta^{(i)}\right)=\log \left(\sum_{Z} P(Y | Z, \theta) P(Z | \theta)\right)-\log P\left(Y | \theta^{(i)}\right)\tag{10}
$$
利用$Jensen$不等式得到下界：
$$
\begin{aligned} L(\theta)-L\left(\theta^{(i)}\right) &=\log \left(\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right)}\right)-\log P\left(Y | \theta^{(i)}\right) \\ & \geqslant \sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right)}-\log P\left(Y | \theta^{(i)}\right) \\ &=\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)} \end{aligned}\tag{11}
$$
令
$$
B\left(\theta, \theta^{(i)}\right) \doteq L\left(\theta^{(i)}\right)+\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)}\tag{12}
$$
则
$$
L(\theta) \geqslant B\left(\theta, \theta^{(i)}\right)\tag{13}
$$
即函数$B\left(\theta, \theta^{(i)}\right)$是$L(\theta)$的一个下界，且由式$(4)$可知，
$$
L\left(\theta^{(i)}\right)=B\left(\theta^{(i)}, \theta^{(i)}\right)\tag{14}
$$
因此当$\theta$使$B\left(\theta, \theta^{(i)}\right)$增大时也可以使$L(\theta)$增大，为了使$L(\theta)$尽可能地增长，选择$\theta^{(i+1)}$使$B\left(\theta, \theta^{(i)}\right)$达到极大，即：
$$
\theta^{(i+1)}=\arg \max _{\theta} B\left(\theta, \theta^{(i)}\right)\tag{15}
$$
带入得：
$$
\begin{aligned} \theta^{(i+1)} &=\arg \max _{\theta}\left(L\left(\theta^{(i)}\right)+\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log \frac{P(Y | Z, \theta) P(Z | \theta)}{P\left(Z | Y, \theta^{(i)}\right) P\left(Y | \theta^{(i)}\right)}\right) \\ &=\arg \max _{\theta}\left(\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log (P(Y | Z, \theta) P(Z | \theta))\right) \\ &=\arg \max _{\theta}\left(\sum_{Z} P\left(Z | Y, \theta^{(i)}\right) \log P(Y, Z | \theta)\right) \\ &=\arg \max _{\theta} Q\left(\theta, \theta^{(i)}\right) \end{aligned}\tag{16}
$$
式$(16)$等价于$EM$的一次迭代，即求$Q$函数及其极大化，$EM$是通过不断求解下界的极大化逼近求解对数似然函数极大化的算法。

**$EM$算法在GMM中的应用**

首先给出高斯分布的概率密度
$$
P(x|\mu_{k}, \sigma_{k})=\frac{1}{\sqrt{2 \pi} \sigma_{k}} \exp \left(-\frac{\left(x-\mu_{k}\right)^{2}}{2 \sigma_{k}^{2}}\right)\\
or\\
P(x|\mu, \Sigma) = \frac{1}{\sqrt{(2 \pi)^{k}|\mathbf{\Sigma}|}} \exp \left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}} \mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)\tag{17}
$$
混合高斯模型具有以下概率密度
$$
P(x | \theta)=\sum_{k=1}^{K} \alpha_{k} P\left(x | \theta_{k}\right)\tag{18}
$$
其中，$\alpha_{k}$是系数，$\alpha_{k} \geqslant 0, \quad \sum_{k=1}^{K} \alpha_{k}=1, \theta_{k}=\left(\mu_{k}, \sigma_{k}^{2}\right)$

假设对于样本生成由高斯混合分布给出：首先根据$\alpha_{1},\alpha_{2},\dotsc ,\alpha_{k}$定义的先验分布选择高斯混合成分，其中$\alpha_{i}$为选择第$i$个混合成分的概率，再根据被选择的混合成分的概率密度来函数进行采样，令随机变量$z_{j}\in \{1, 2, \dotsc , k\}$表示生成样本$x_{j}$的高斯混合成分，显然$z_{j}$的先验分布$P(z_{j}=i)$对应于$\alpha_{i} (i=1, 2, \dotsc , k)$，根据贝叶斯定理$z_{j}$的后验分布为
$$
\begin{aligned} P\left(z_{j}=i | x_{j}\right) &=\frac{P\left(z_{j}=i\right) \cdot P\left(x_{j} | z_{j}=i\right)}{P\left(x_{j}\right)} \\ &=\frac{\alpha_{i} \cdot p\left(x_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(x_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \end{aligned}\tag{19}
$$
将$P\left(z_{j}=i | x_{j}\right)$记为$\gamma_{ji}(i=1, 2, \dotsc , k)$，那么对于式$(18)$，模型参数$\left\{\left(\alpha_{i}, \boldsymbol{\mu}_{i}, \Sigma_{i}\right) | 1 \leqslant i \leqslant k\right\}$的求解可以采用极大对数自然，即最大化对数自然函数
$$
\begin{aligned} L L(X) &=\ln \left(\prod_{j=1}^{m} P\left(x_{j}\right)\right) \\ &=\sum_{j=1}^{m} \ln \left(\sum_{i=1}^{k} \alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)\right) \end{aligned}\tag{20}
$$
若能使式$(18)$最大，则由$\frac{\partial L L(D)}{\partial \mu_{i}}=0$可得
$$
\begin{aligned} \frac{\partial L L(X)}{\partial \boldsymbol{\mu}_{i}} &=\frac{\partial}{\partial \boldsymbol{\mu}_{i}}\left[\sum_{j=1}^{m} \ln \left(\sum_{i=1}^{k} \alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)\right] \\ &=\sum_{j=1}^{m} \frac{\partial}{\partial \boldsymbol{\mu}_{i}}\left[\ln \left(\sum_{i=1}^{k} \alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)\right] \\ &=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot \frac{\partial}{\partial \boldsymbol{\mu}_{i}}\left(P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \boldsymbol{\Sigma}_{l}\right)}\\ &=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot \frac{1}{(2 \pi)^{\frac{n}{2}}\left|\mathbf{\Sigma}_{i}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \mathbf{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \frac{\partial}{\partial \boldsymbol{\mu}_{i}}\left(-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \mathbf{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\right)\\&=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \cdot\left(-\frac{1}{2}\right) \cdot \frac{\partial}{\partial \boldsymbol{\mu}_{i}}\left(\boldsymbol{x}_{j}^{T} \mathbf{\Sigma}_{i}^{-1} \boldsymbol{x}_{j}-\boldsymbol{x}_{j}^{T} \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{i}^{T} \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{x}_{j}+\boldsymbol{\mu}_{i}^{T} \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}\right)\\&=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \cdot\left(-\frac{1}{2}\right) \cdot \frac{\partial}{\partial \boldsymbol{\mu}_{i}}\left(-\boldsymbol{x}_{j}^{T} \mathbf{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}-\boldsymbol{\mu}_{i}^{T} \mathbf{\Sigma}_{i}^{-1} \boldsymbol{x}_{j}+\boldsymbol{\mu}_{i}^{T} \mathbf{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}\right)\\&=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \cdot\left(-\frac{1}{2}\right) \cdot \frac{\partial}{\partial \boldsymbol{\mu}_{i}}\left(-2 \boldsymbol{\mu}_{i}^{T} \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{x}_{j}+\boldsymbol{\mu}_{i}^{T} \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}\right)\\&=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \cdot\left(-\frac{1}{2}\right) \cdot\left(-2 \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{x}_{j}+2 \boldsymbol{\Sigma}_{i}^{-1} \boldsymbol{\mu}_{i}\right)\\&=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \mathbf{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\end{aligned}\tag{21}
$$
令上式等于$0$，左右两边同时乘于$\boldsymbol{\Sigma}_{i}$有：
$$
\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)=0\tag{22}
$$
由式$(19)$以及$\gamma_{j i}=P\left(z_{j}=i | \boldsymbol{x}_{j}\right)$有
$$
\boldsymbol{\mu}_{i}=\frac{\sum_{j=1}^{m} \gamma_{j i} \boldsymbol{x}_{j}}{\sum_{j=1}^{m} \gamma_{j i}}
$$
由$\frac{\partial L L(X)}{\partial \boldsymbol{\Sigma}_{i}}=0$有
$$
\begin{aligned} \frac{\partial L L(X)}{\partial \boldsymbol{\Sigma}_{i}} &=\frac{\partial}{\partial \boldsymbol{\Sigma}_{i}}\left[\sum_{j=1}^{m} \ln \left(\sum_{i=1}^{k} \alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)\right] \\ &=\sum_{j=1}^{m} \frac{\partial}{\partial \boldsymbol{\Sigma}_{i}}\left[\ln \left(\sum_{i=1}^{k} \alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)\right] \\ &=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot \frac{\partial}{\partial \boldsymbol{\Sigma}_{i}}\left(P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \end{aligned}\tag{23}
$$
其中
$$
\begin{aligned} \frac{\partial}{\partial \boldsymbol{\Sigma}_{i}}\left(P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)\right) &=\frac{\partial}{\partial \boldsymbol{\Sigma}_{i}}\left[\frac{1}{(2 \pi)^{\frac{n}{2}}\left|\boldsymbol{\Sigma}_{i}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\right)\right] \\ &=\frac{\partial}{\partial \boldsymbol{\Sigma}_{i}}\left\{\exp \left[\ln \left(\frac{1}{(2 \pi)^{\frac{n}{2}}\left|\boldsymbol{\Sigma}_{i}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\right)\right)\right]\right\}\\&=P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot\left[-\frac{1}{2} \frac{\partial\left(\ln \left|\mathbf{\Sigma}_{i}\right|\right)}{\partial \boldsymbol{\Sigma}_{i}}-\frac{1}{2} \frac{\partial\left[\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\right]}{\partial \boldsymbol{\Sigma}_{i}}\right]\\&=P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right) \cdot\left[-\frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1}+\frac{1}{2} \boldsymbol{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \boldsymbol{\Sigma}_{i}^{-1}\right] \end{aligned}\tag{24}
$$
带回$\frac{\partial L L(X)}{\partial \boldsymbol{\Sigma}_{i}}$有
$$
\frac{\partial L L(X)}{\partial \boldsymbol{\Sigma}_{i}}=\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \cdot\left[-\frac{1}{2} \mathbf{\Sigma}_{i}^{-1}+\frac{1}{2} \mathbf{\Sigma}_{i}^{-1}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T} \boldsymbol{\Sigma}_{i}^{-1}\right]\tag{25}
$$
将$\frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}=\gamma_{j i}$带入等于0可得
$$
\sigma_{k}^{2}=\frac{\sum_{j=1}^{m} \gamma_{j i}\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)\left(\boldsymbol{x}_{j}-\boldsymbol{\mu}_{i}\right)^{T}}{\sum_{j=1}^{m} \gamma_{j i}}
$$
对于$\alpha_{i}$，考虑$LL(X)$的拉格朗日形式
$$
L L(X)+\lambda\left(\sum_{i=1}^{k} \alpha_{i}-1\right)\tag{26}
$$
对于式$(26)$对$\alpha_{i}$导数为0，有
$$
\sum_{j=1}^{m} \frac{P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}+\lambda=0\tag{27}
$$
两边同时乘于$\alpha_{i}$有
$$
\begin{array}{l}{\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}+\lambda \alpha_{i}=0} \\ {\sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}=-\lambda \alpha_{i}}\end{array}
$$
两边对所有的$\alpha$求和可得
$$
\begin{aligned} \sum_{i=1}^{k} \sum_{j=1}^{m} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} &=-\lambda \sum_{i=1}^{k} \alpha_{i} \\ \sum_{j=1}^{m} \sum_{i=1}^{k} \frac{\alpha_{i} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{i}, \boldsymbol{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot P\left(\boldsymbol{x}_{j} | \boldsymbol{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} &=-\lambda \sum_{i=1}^{k} \alpha_{i} \\ m =-\lambda \end{aligned}
$$
因此有
$$
\alpha_{i}=\frac{1}{m} \sum_{j=1}^{m} \gamma_{j i}
$$

##### 2. 算法的实现

###### k-means

输入：$n$个样本的集合$X$

输出：样本集合的聚类$C^{*}$

1. 初始化。令$t=0$，随机选择$k$个样本点作为初始聚类中心$m^{(0)}=(m_{1}^{(0)}, \dotsc ,m_{l}^{(0)},\dotsc ,m_{k}^{(0)})$。
2. 对样本中心进行聚类。对固定的类中心$m^{(t)}=(m_{1}^{(t)}, \dotsc ,m_{l}^{(t)},\dotsc ,m_{k}^{(t)})$，其中$m_{l}^{(t)}$为类$G_{l}$的中心，计算每个样本到类中心的距离，将每个样本指派到与其最近的中心的类中，构成聚类结果$C^{(t)}$。
3. 计算新的类中心，对聚类结果$C^{(t)}$，计算当前各个类中的样本的均值，作为新的类中心$m^{(t+1)}=(m_{1}^{(t+1)}, \dotsc ,m_{l}^{(t+1)},\dotsc ,m_{k}^{(t+1)})$。
4. 如果迭代收敛或符合停止条件，输出$C^{*}=C^{(t)}$

###### EM​

输入：观测数据$x_{1},x_{2},\dotsc ,x_{N}$,高斯混合模型

输出：高斯混合模型参数

1. 取参数的初始值开始迭代

2. $E$步：依据当前模型参数，计算分模型$k$对观测数据$x_{i}$的响应度
   $$
   \hat{\gamma}_{j k}=\frac{\alpha_{k} P\left(x_{j} | \theta_{k}\right)}{\sum_{k=1}^{k} \alpha_{k} P\left(x_{j} | \theta_{k}\right)}, \quad j=1,2, \cdots, N ; k=1,2, \cdots, K
   $$
   
3. $M$步：计算进行新一轮迭代的模型参数
   $$
   \begin{aligned} \hat{\mu}_{k}&=\frac{\sum_{j=1}^{N} \hat{r}_{j k} x_{j}}{\sum_{j=1}^{N} \hat{\gamma}_{j k}}, & k=1,2, \cdots, K \\ \hat{\sigma}_{k}^{2}&=\frac{\sum_{j=1}^{N} \hat{\gamma}_{j k}\left(y_{j}-\mu_{k}\right)^{2}}{\sum_{j=1}^{N} \hat{\gamma}_{j k}}, & k=1,2, \cdots, K\\ \hat{\alpha}_{k}&=\frac{\sum_{j=1}^{N} \hat{y}_{j k}}{N}, & k=1,2, \cdots, K \end{aligned}
   $$
4. 重复第2和第3步，直到收敛
#### 四、实验结果与分析

##### k-means

![1](D:\Document\CS doc\machine learning\lab3\1.png)

初始随机选点，进行了25实验发现聚类效果均很好，未出现$EM$算法，初始数据不好导致最后分类效果不好的情况
##### EM

![2](D:\Document\CS doc\machine learning\lab3\2.png)
初始随机选点，可以看到和$k-means$的聚类效果相似均很好，因为初始随机，在25次实验中出现了下图所示情况

![3](D:\Document\CS doc\machine learning\lab3\3.png)
但是由于出现概率很低，因此，认为仍可以随机初始化。
##### uci数据集

使用的是种子数据，是一个7维三分类数据，数据中第8维为已有标签，根据已有标签和分类结果得到准确率如下图所示

![4](D:\Document\CS doc\machine learning\lab3\4.png)

#### 五、结论
1. 对于生成的数据$k-means$和$EM$的效果都很好，认为主要原因是生成的高斯分布较好，对于uci数据分类效果也较为理想，但是和生成数据比准确率下降；
2. $k-means$的初始化，最好选取$k$个距离较远的点最为初始质心进行迭代，以防止最终聚类效果不好，陷入不好的局部最优化；
3. $EM$同$k-means$一样，得到的结果仍旧是局部最优解，可以对参数选取不同的初始值多次计算。


#### 六、参考文献

#### 七、附录：源代码（带注释）

lab3_kmeans.py

```python
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
```

lab3_em.py

```python
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
    scale_of_example = 100
    dimensionension = 2
    mu1 = np.array([[0, 4]])
    sigma1 = np.array([[2, 0], [0, 2]])
    R1 = cholesky(sigma1)
    x1 = np.dot(np.random.randn(scale_of_example, dimensionension), R1) + mu1
    mu2 = np.array([[4, 6]])
    sigma2 = np.array([[2, 0], [0, 2]])
    R2 = cholesky(sigma2)
    x2 = np.dot(np.random.randn(scale_of_example, dimensionension), R2) + mu2
    mu3 = np.array([[2, -2]])
    sigma3 = np.array([[2, 0], [0, 2]])
    R3 = cholesky(sigma3)
    x3 = np.dot(np.random.randn(scale_of_example, dimensionension), R3) + mu3
    cmp = np.vstack((mu1, mu2))
    cmp = np.vstack((cmp, mu3))

    data0 = np.vstack((x1, x2))
    data = np.vstack((data0, x3))
    label0 = np.vstack(
        (
            np.ones((scale_of_example, 1)),
            np.ones((scale_of_example, 1)) + np.ones((scale_of_example, 1)),
        )
    )
    label = np.vstack(
        (
            label0,
            np.ones((scale_of_example, 1))
            + np.ones((scale_of_example, 1))
            + np.ones((scale_of_example, 1)),
        )
    )
    print("The EM performance on the generated GMM")
    accuracy, type0, n0, type1, n1, type2, n2, mu = EM(data, 3, 300, 2, label)
    print("The accuracy :")
    print(accuracy)
    print("visualization")
    plt.plot(type0[0:n0, 0], type0[0:n0, 1], ".", color="red")
    plt.plot(type1[0:n1, 0], type1[0:n1, 1], ".", color="blue")
    plt.plot(type2[0:n2, 0], type2[0:n2, 1], ".", color="green")
    plt.scatter(mu[:, 0], mu[:, 1], color="black")
    plt.show()
    print()
    print("The EM performance on the uci data set")
    uci_data = np.loadtxt("seeds_dataset.txt")
    uci_data = np.mat(uci_data)
    accuracy, type0, n0, type1, n1, type2, n2, mu = EM(uci_data[:, :7], 3, 210, 7, np.mat(uci_data[:, 7]))
    print("The accuracy:")
    print(accuracy)
```

