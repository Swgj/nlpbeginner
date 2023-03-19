import numpy as np
import string
from collections import Counter

class Preprocessor(object):
    def __init__(self,
                 lowercase = False,
                 ignore_punctuation=False,
                 num_words = None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos

    def read_data(self,filepath):
        with open(filepath,"r",encoding="utf-8") as input_data:
            ids,premises,hypotheses,labels = [],[],[],[]

            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})
            # 去掉文件第一行的标题
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")
                if line[0] =='-': continue

                pair_id = line[7]
                premise = line[1]
                hypothesis = line[2]

                #去掉括号()
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                labels.append(line[0])
                ids.append(pair_id)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}

    def build_worddict(self, data):
        """构建词典"""
        words = []
        [words.extend(sentence) for sentence in data["premises"]]
        [words.extend(sentence) for sentence in data["hypotheses"]]

        #计数
        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}
        #用于填充 padding
        self.worddict["_PAD_"] = 0
        #out-of-vocabulary
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset +=1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset +=1

        for i,word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i +offset

        if self.labeldict =={}:
            label_names = set(data["labels"])
            self.labeldict = {label_name :i
                              for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """句子转向量"""
        indices = []
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                index = self.worddict["_OOV_"]
            indices.append(index)

        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        "向量转句子"
        return [
            list(self.worddict.keys())[
                list(self.worddict.values()).index(i)]
                for i in indices]

    def transform_to_indices(self,data):
        """将dataset中，premises、hypotheses、labels转为向量"""
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "labels": []}

        for i, premise in enumerate(data["premises"]):
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["ids"].append(data["ids"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def build_embedding_matrix(self,embeddings_file):
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix