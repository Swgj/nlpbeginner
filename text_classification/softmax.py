#softmax函数
import numpy as np
import random

class Softmax:
    """Softmax"""
    def __init__(self,train_size,type_size,feature):
        """
        :param train_size: 训练集样本个数
        :param type_size: 分类种类数
        :param feature: 向量长度
        """
        self.train_size = train_size
        self.type_size = type_size
        self.feature = feature
        self.weight = np.random.randn(feature,type_size) #参数矩阵初始化

    def softmax_calculate(self,x):
        """
        计算向量的softmax值
        :param x: 向量
        :return: softmax value
        """
        exp = np.exp(x-np.max(x)) #防止指数太大溢出
        return exp/exp.sum()

    def softmax_all(self,wtx):
        """计算矩阵的softmax值"""
        wtx -= np.max(wtx,axis=1,keepdims=True)
        wtx = np.exp(wtx)
        wtx/=np.sum(wtx,axis=1,keepdims=True)
        return wtx

    def magic_y(self,y):
        """将种类转为one-hot向量"""
        res = np.array([0]*self.type_size)
        res[y] = 1
        return res.reshape(-1,1) #1xn的向量转为nx1的向量

    def predict(self,x):
        """对输入的向量x，判断句子的种类分布概率"""
        prob = self.softmax_all(x.dot(self.weight)) #向量乘法
        return prob.argmax(axis=1) #将概率最大的位置输出--种类

    def accuracy(self,train,train_y,test,test_y):
        """计算准确率Accuracy"""
        train_size = len(train)
        train_prediction = self.predict(train)
        train_accuracy = sum([train_y[i]==train_prediction[i] for i in range(train_size)])/train_size

        test_size = len(test)
        test_prediction = self.predict(test)
        test_accuracy = sum([test_y[i]==test_prediction[i] for i in range(test_size)])/test_size
        print(train_accuracy,test_accuracy)
        return train_accuracy,test_accuracy

    def regression(self,x,y,alpha,epoches,strategy="mini",mini_size=100):
        """softmax regression"""
        if self.train_size != len(x) or self.train_size != len(y):
            raise Exception("样本个数不匹配！")
        if strategy == "mini": #mini batch策略
            for i in range(epoches):
                increment = np.zeros((self.feature,self.type_size)) #梯度矩阵初始化
                for j in range(mini_size): #随机抽样k次
                    k = random.randint(0,self.type_size-1)
                    y_hat = self.softmax_calculate(self.weight.T.dot(x[k].reshape(-1,1)))#预测值
                    increment += x[k].reshape(-1,1).dot((self.magic_y(y[k])-y_hat).T) #计算梯度
                self.weight += alpha/mini_size * increment #更新参数
        elif strategy == "shuffle": #随机梯度下降
            for i in range(epoches):
                k = random.randint(0,self.train_size-1) #抽取一个
                y_hat = self.softmax_calculate(self.weight.T.dot(x[k].reshape(-1,1)))
                increment = x[k].reshape(-1,1).dot((self.magic_y(y[k]) - y_hat).T)
                self.weight += alpha*increment #更新参数
        elif strategy=="batch": #整批量梯度
            for i in range(epoches):
                increment = np.zeros((self.feature,self.type_size))
                for j in range(self.train_size):
                    y_hat = self.softmax_calculate(self.weight.T.dot(x[j].reshape(-1,1)))
                    increment += x[j].reshape(-1,1).dot((self.magic_y(y[j]) - y_hat).T)
                self.weight += alpha/self.train_size * increment
        else:
            raise Exception("无此策略")