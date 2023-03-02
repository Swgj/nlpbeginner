import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_filters, filter_sizes,output_dim,dropout):
        '''
        :param vocab_size: 词典大小
        :param embedding_dim: 对于每个词，word embedding向量长度
        :param num_filters: 卷积核的数量
        :param filter_sizes: 卷积核的大小
        :param output_dim: 卷积后模型输出维度
        :param dropout: dropout的比例
        '''
        super().__init__()
        # 1.嵌入层
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        # 2.卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,out_channels=num_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes
        ])
        # 3.全连接层
        self.full_connected = nn.Linear(len(filter_sizes)*num_filters,output_dim)
        # 4.Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        '''
        前向传播
        :param text: 用于判断的文本
        :return:
        '''
        embedded = self.embedding(text)
        #为了与卷积层格式匹配，需要在嵌入向量的第二个维度上添加一个维度
        embedded = embedded.unsqueeze(1)
        #卷积操作
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #对卷积结果最大池化操作
        pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]
        #将不同卷积核的池化结果拼接起来
        cat = self.dropout(torch.cat(pooled, dim=1))
        #将拼接的结果输入全连接层进行分类
        return self.full_connected(cat)