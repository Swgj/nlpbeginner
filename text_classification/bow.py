# Bag of words
class Bow:
    """词袋模型"""
    def __init__(self,data):
        self.data = data
        self.word_dict = dict() #词典
        self.dict_size = 0 #词典大小

    def transfer_words(self):
        """将data中的词转为向量"""
        for term in self.data:
            sentence = term[2]
            sentence = sentence.lower() #转小写
            words = sentence.split() #按照空格分词
            for word in words:
                if word not in self.word_dict:#将未出现的词填入词典
                    self.word_dict[word] = len(self.word_dict)
        self.dict_size = len(self.word_dict) #更新词典大小

