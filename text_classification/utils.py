import numpy as np
import matplotlib.pyplot as plt
from softmax import Softmax

def data_split(datas,train_size =0.7,max_size=1000):
    """
    将数据划分为训练集和测试集
    :param datas: 数据
    :param train_size: 训练集比例
    :param max_size: 数据集的最大数量--不设置会因数据太多导致内存溢出！！
    :return: train_set,test_set: 训练集和测试集
    """
    np.random.seed(2023)
    data_size = len(datas)
    use_size = min(data_size,max_size) #选取不超过max_size条数据
    train_indices = np.random.choice(use_size,int(use_size*train_size),replace=False)
    test_indices = np.array(list(set(range(use_size))-set(train_indices)))
    train_set = np.array(datas)[train_indices]
    test_set = np.array(datas)[test_indices]
    return train_set,test_set

#训练-绘图
def model_plot(model,epoches,mini_size):
    alphas = [0.001,0.01,0.1,1,10,100,1000,10000]

    #mini-batch
    mini_train = list()
    mini_test = list()
    for alpha in alphas:
        soft = Softmax(len(model.train_set),5,model.diction_size)
        soft.regression(model.train_matrix,model.train_y,alpha,int(epoches/mini_size),"mini",mini_size)
        r_train,r_test = soft.accuracy(model.train_matrix,model.train_y,model.test_matrix,model.test_y)
        mini_train.append(r_train)
        mini_test.append(r_test)

    #绘制结果图
    plt.semilogx(alphas,mini_train,'r--',label='train') #横坐标采用对数格式
    plt.semilogx(alphas,mini_test,'b--',label="test")
    plt.title("N-gram")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.show()
