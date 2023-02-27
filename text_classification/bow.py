from utils import data_split
import numpy as np

# Bag of words
class Bow:
    """词袋模型"""
    def __init__(self,data,max_size=1000):
        self.data = data[:max_size]
        self.maxsize = max_size
        self.word_dict = dict() #词典
        self.dict_size = 0 #词典大小
        self.train,self.test = data_split(data,train_size=0.7,max_size=max_size)
        self.train_y = [int(term[3]) for term in self.train] #训练集y值
        self.test_y = [int(term[3]) for term in self.test] #测试集y值
        self.train_matrix = None
        self.test_matrix = None

    def get_words(self):
        for term in self.data:
            s = term[2]
            s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
            words = s.split()
            for word in words:  # 一个一个单词寻找
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
        self.len = len(self.word_dict)
        self.test_matrix = np.zeros((len(self.test), self.len))  # 初始化0-1矩阵
        self.train_matrix = np.zeros((len(self.train), self.len))  # 初始化0-1矩阵

    def get_matrix(self):
        for i in range(len(self.train)):  # 训练集矩阵
            s = self.train[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.train_matrix[i][self.word_dict[word]] = 1
        for i in range(len(self.test)):  # 测试集矩阵
            s = self.test[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.test_matrix[i][self.word_dict[word]] = 1
