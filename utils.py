import codecs
from keras_bert import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm

maxlen = 180
embedding_size = 300
#bert_path
# config_path = '/home/ccit22/m_minbo/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/home/ccit22/m_minbo/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/home/ccit22/m_minbo/chinese_L-12_H-768_A-12/vocab.txt'
config_path = '/home/ccit/tkhoon/baiduie/sujianlin/myself_model/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/ccit/tkhoon/baiduie/sujianlin/myself_model/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/ccit/tkhoon/baiduie/sujianlin/myself_model/bert/chinese_L-12_H-768_A-12/vocab.txt'

def seq_padding(X):
    # L = [len(x) for x in X]
    # ML =
    return [x + [0] * (maxlen - len(x)) if len(x) < maxlen else x[:maxlen] for x in X]

def encode(text):
    vocabs = set()
    with open(dict_path, encoding='utf8') as f:
        for l in f:
            vocabs.add(l.replace('\n', ''))

    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    tokens = ['[CLS]'] + [ch if ch in vocabs else '[UNK]' for ch in text] + ['[SEP]']
    return tokenizer._convert_tokens_to_ids(tokens), [0] * len(tokens)


def load_data(data,mode):
    """
    mode: dev ，同时还要返回bio
    mode: test , 只返回序列化文本
    :param data:
    :param mode:
    :return:
    """
    if mode == 'dev':
        idxs = [i for i in range(len(data))]
        BERT_INPUT0, BERT_INPUT1,BIO= [], [], []
        for i in tqdm(idxs):
            d = data[i]
            text = d['text']
            or_text = text
            indices, segments = encode(or_text)
            entity = d['entity']
            text = '^' + text + '^'
            bio = get_data_bio(text, entity)
            BERT_INPUT0.append(indices)
            BERT_INPUT1.append(segments)
            BIO.append(bio)
        BERT_INPUT0 = np.array(seq_padding(BERT_INPUT0))
        BERT_INPUT1 = np.array(seq_padding(BERT_INPUT1))
        BIO = np.array(seq_padding(BIO))
        return BERT_INPUT0, BERT_INPUT1,BIO
    else:
        idxs = [i for i in range(len(data))]
        BERT_INPUT0, BERT_INPUT1= [], []
        for i in tqdm(idxs):
            d = data[i]
            text = d['text']
            or_text = text
            indices, segments = encode(or_text)
            BERT_INPUT0.append(indices)
            BERT_INPUT1.append(segments)
        BERT_INPUT0 = np.array(seq_padding(BERT_INPUT0))
        BERT_INPUT1 = np.array(seq_padding(BERT_INPUT1))
        return BERT_INPUT0, BERT_INPUT1


def get_data_bio(text,entity):
    """
    bio, 0 ,1 ,2
    :param text:
    :return:
    """
    bio_list = [0]*len(text)
    start_index = text.find(entity)
    end_index = start_index + len(entity) - 1
    bio_list[start_index] = 1
    if start_index == -1:
        return bio_list
    else:
        for i in range(start_index+1,end_index+1):
            bio_list[i] = 2
    return bio_list


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.tokenizer = Tokenizer(self.token_dict)
        self.cache_data = []
        self.vocabs = set()
        with open(dict_path, encoding='utf8') as f:
            for l in f:
                self.vocabs.add(l.replace('\n', ''))

    def init_cache_data(self):
        cur_step = 0
        for i, t in enumerate(self.get_next()):
            if i >= self.steps:
                break
            cur_step += 1
            self.cache_data.append(t)

    def __len__(self):
        return self.steps

    def encode(self, text):
        tokens = ['[CLS]'] + [ch if ch in self.vocabs else '[UNK]' for ch in text] + ['[SEP]']
        return self.tokenizer._convert_tokens_to_ids(tokens), [0] * len(tokens)
    def __iter__(self):
        while True:
            idxs = [i for i in range(len(self.data))]
            np.random.shuffle(idxs)
            BERT_INPUT0, BERT_INPUT1,BIO = [],[],[]
            for i in idxs:
                d = self.data[i]
                text = d['text']
                or_text = text
                indices, segments = self.encode(or_text)
                entity = d['entity']
                text = '^' + text + '^'
                bio = get_data_bio(text, entity)
                BERT_INPUT0.append(indices)
                BERT_INPUT1.append(segments)
                BIO.append(bio)
                if len(BERT_INPUT1) == self.batch_size or i == idxs[-1]:
                    BERT_INPUT0 = np.array(seq_padding(BERT_INPUT0))
                    BERT_INPUT1 = np.array(seq_padding(BERT_INPUT1))
                    BIO = np.array(seq_padding(BIO))
                    yield [BERT_INPUT0, BERT_INPUT1,BIO], None
                    BERT_INPUT0, BERT_INPUT1,BIO= [],[],[]