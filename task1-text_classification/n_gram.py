import numpy as np
from utils import data_split

#N-gram
class N_Gram:
    """N元语法模型"""
    def __init__(self,data,n=2,max_size=1000):
        """
        :param data: 原始数据集
        :param n: n元语法
        :param max_size: 最大数据量--防止内存溢出
        """
        self.data = data[:max_size] #截取最大数据量
        self.n = n
        self.max_size = max_size
        self.diction_size = 0 #特征数量
        self.words_diction = dict() #词典
        self.train_set,self.test_set = data_split(data,train_size=0.7,max_size=self.max_size)
        self.train_y = [int(senten[3])for senten in self.train_set]
        self.test_y = [int(senten[3])for senten in self.test_set]
        self.train_matrix = np.zeros((len(self.train_set),self.diction_size))#初始化训练集向量
        self.test_matrix = np.zeros((len(self.test_set),self.diction_size)) #初始化测试集向量

    def get_words(self):
        for demention in range(1,self.n+1): #获取所有的1元短语到n元短语
            for data_line in self.data:
                senten = data_line[2]
                senten = senten.lower()
                words = senten.split()
                for i in range(len(words)-demention+1):
                    tmp = words[i:i+demention]
                    tmp = "_".join(tmp) #连接
                    if tmp not in self.words_diction:
                        self.words_diction[tmp] = len(self.words_diction) #填入词典
        self.diction_size = len(self.words_diction)
        #更新矩阵大小
        self.train_matrix = np.zeros((len(self.train_set),self.diction_size))
        self.test_matrix = np.zeros((len(self.test_set),self.diction_size))

    def get_matrix(self):
        for demention in range(1,self.n+1):
            for i in range(len(self.train_set)):
                sen = self.train_set[i][2] #一条文本数据
                sen = sen.lower()
                words = sen.split()
                for j in range(len(words) - demention +1):
                    tmp = words[j:j+demention]
                    tmp = "_".join(tmp)
                    self.train_matrix[i][self.words_diction[tmp]]=1 #设置向量的值
            for i in range(len(self.test_set)):
                sen = self.test_set[i][2]
                sen = sen.lower()
                words = sen.split()
                for j in range(len(words) -demention+1):
                    tmp = words[j:j+demention]
                    tmp = "_".join(tmp)
                    self.test_matrix[i][self.words_diction[tmp]]=1 #设置测试向量的值