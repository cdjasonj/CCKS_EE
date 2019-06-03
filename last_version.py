import numpy as np
import codecs
import json
import os
# import deepcopy
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from random import choice, sample
from keras_bert import Tokenizer
from keras_bert import get_model, load_model_weights_from_checkpoint
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from layers import Gate_Add_Lyaer,MaskedConv1D,MaskedLSTM,MaskFlatten,MaskPermute,MaskRepeatVector

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

config_path = '/home/ccit/tkhoon/baiduie/sujianlin/myself_model/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/ccit/tkhoon/baiduie/sujianlin/myself_model/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/ccit/tkhoon/baiduie/sujianlin/myself_model/bert/chinese_L-12_H-768_A-12/vocab.txt'
train_data_path = 'input/train_data_me.json'
test_data_path = 'input/dev_data_me.json'
# test_data_path = 'input/event_type_entity_extract_eval.csv'

#input_path
event2id_path ='input/event2id.json'
char2id_path = 'input/all_chars_me.json'
origin_test_data_path = "input/event_type_entity_extract_eval.csv"
#output_path
weight_name=  'models/baseline_bert_522.weights'
# dev_result = 'version1_dev16.json'
test_result_path = 'outputs'

char_size = 128
debug = True

id2char,char2id = json.load(open(char2id_path,encoding='utf-8'))
event2id = json.load(open(event2id_path,encoding='utf-8'))
_train_data = json.load(open(train_data_path,encoding='utf-8'))
test_data = json.load(open(test_data_path,encoding='utf-8'))
D = pd.read_csv('input/event_type_entity_extract_eval.csv', encoding='utf-8', header=None)

if debug==True:
    _train_data = _train_data[:2000]

maxlen = 180
embedding_size = 300
hidden_size = 128
vocab_size = len(char2id)+2 #0pad 1 UNK
event_size = len(event2id)
event_embedding_size = 30

def seq_padding(X):
    # L = [len(x) for x in X]
    # ML =
    return [x + [0] * (maxlen - len(x)) if len(x) < maxlen else x[:maxlen] for x in X]

def split_data(_train_data):
    all_data = []
    for data in _train_data:
        if data['event_type'] != '其他':
            all_data.append(data)
    #随机取样0.2作为验证集
    dev_data = sample(all_data,round(len(all_data)*0.2))
    train_data = []
    for data in all_data:
        flag = 0
        text = data['text']
        for _data in dev_data:
            if _data['text'] ==text:
                flag = 1
                break
        if flag == 0 :
            train_data.append(data)
    return train_data,dev_data


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


def build_model_from_config(config_file,
                            checkpoint_file,
                            training=False,
                            trainable=False,
                            seq_len=None,
                            ):
    """Build the model from config file.

    :param config_file: The path to the JSON configuration file.
    :param training: If training, the whole model will be returned.
    :param trainable: Whether the model is trainable.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: model and config
    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())
    if seq_len is not None:
        config['max_position_embeddings'] = min(seq_len, config['max_position_embeddings'])
    if trainable is None:
        trainable = training
    model = get_model(
        token_num=config['vocab_size'],
        pos_num=config['max_position_embeddings'],
        seq_len=config['max_position_embeddings'],
        embed_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        feed_forward_dim=config['intermediate_size'],
        training=False,
        trainable=True,
    )
    inputs, outputs = model
    bio_label = Input(shape=(maxlen,))

    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(inputs[0])

    attention = TimeDistributed(Dense(1, activation='tanh'))(outputs)
    attention = MaskFlatten()(attention)
    attention = Activation('softmax')(attention)
    attention = MaskRepeatVector(config['hidden_size'])(attention)
    attention = MaskPermute([2, 1])(attention)
    sent_representation = multiply([outputs, attention])
    attention = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    # lstm_attention = Lambda(seq_and_vec, output_shape=(None, self.hidden_size * 2))(
    #     [lstm, attention])  # [这里考虑下用相加的方法，以及门控相加]
    attention = MaskRepeatVector(maxlen)(attention)  # [batch,sentence,hidden_size]
    gate_attention = Gate_Add_Lyaer()([outputs, attention])
    gate_attention = Dropout(0.15)(gate_attention)

    cnn1 = MaskedConv1D(filters=hidden_size, kernel_size=3, activation='relu', padding='same')(gate_attention)
    #BIOE
    bio_pred = Dense(4, activation='softmax')(cnn1)

    entity_model = keras.models.Model([inputs[0], inputs[1]], [bio_pred])  # 预测subject的模型
    train_model = keras.models.Model([inputs[0], inputs[1],bio_label],[bio_pred])

    loss = K.sparse_categorical_crossentropy(bio_label, bio_pred)
    loss = K.sum(loss * mask[:, :, 0]) / K.sum(mask)

    train_model.add_loss(loss)
    train_model.summary()
    train_model.compile(
        optimizer=keras.optimizers.Adam(lr=3e-5),
    )
    load_model_weights_from_checkpoint(train_model, config, checkpoint_file, training)
    return train_model,entity_model

def extract_entity(bio_pred,data):
    entities=[]
    text_in = []
    for _data in data:
        text = _data['text']
        text = '^' + text + '^'
        text_in.append(text)
    bio_pred = np.argmax(bio_pred,axis=-1) #[batch_size,sequence_length]
    for idx in range(len(data)):
        text = text_in[idx]
        bio = bio_pred[idx]
        for i in range(len(bio)):
            flag = 0
            if bio[i] == 1: #找到B标识符1
                if i >= len(text): #预测出的1在文本范围外，来自pad的部分
                    break
                else:
                    entity = text[i]
                    flag = 1
                    for j in range(i+1,len(bio)):
                        if j >= len(text): #预测出的结构在文本范围外
                            entities.append(entity)
                            break
                        else:
                            if bio[j] == 2: #找到I标志符2
                                entity+=text[j]
                            else:
                                flag = 1
                                entities.append(entity)
                                break
                    break
        if flag == 0:#没有预测出来补个空
            entities.append('')
    return entities

def comput_f1(entities):
    """
    对dev进行测评
    P_all = right_num_all / pred_num_all  # 准确率
    R_all = right_num_all / true_num_all  # 召回率
    F_all = 2 * P_all * R_all / (P_all + R_all)  # F值
    :param dev_file:
    :return:
    """
    # import ipdb
    # ipdb.set_trace()
    right = 1e-10
    pred = 1e-10
    true = 1e-10
    for idx in range(len(dev_data)):
        pred_entity = entities[idx]
        true_entity = dev_data[idx]['entity']
        true += 1
        if pred_entity:
            pred+= 1
            if pred_entity == true_entity:
                right+=1
    P = right/pred
    R = right / true
    F = 2*P*R/(R+P)
    return P,R,F

def save_result(data,entities):
    """
    按照  id , entites 写入文档
    :param data:
    :param entities:
    :return:
    """
    with open(test_result_path,'w',encoding='utf-8') as fr:
        for index in range(len(data)):
            id = data[index]['id']
            entity = entities[index]
            fr.write(str(id)+','+str(entity)+'\n')

def predict_test_batch(mode):
    if mode == 'test':
        pass
        weight_file = weight_name
        train_model.load_weights(weight_file)
        test_BERT_INPUT0, test_BERT_INPUT1 = load_data(test_data,'test')
        bio_pred =entity_model.predict([test_BERT_INPUT0, test_BERT_INPUT1],batch_size=1000,verbose=1) #[batch_size,sentence,num_classes]
        entites = extract_entity(bio_pred,test_data)
        save_result(test_data,entites)
    else:
        #对dev进行测评
        dev_BERT_INPUT0, dev_BERT_INPUT1,_ = load_data(dev_data,'dev')
        bio_pred =entity_model.predict([dev_BERT_INPUT0, dev_BERT_INPUT1],batch_size=1000,verbose=1) #[batch_size,sentence,num_classes]
        entites = extract_entity(bio_pred,dev_data)
        return comput_f1(entites)

def scheduler(epoch):
    # 每隔1个epoch，学习率减小为原来的1/2
    # if epoch % 100 == 0 and epoch != 0:
    #再epoch > 3的时候,开始学习率递减,每次递减为原来的1/2,最低为2e-6
    if epoch+1 <= 2:
        return K.get_value(train_model.optimizer.lr)
    else:
        lr = K.get_value(train_model.optimizer.lr)
        lr = lr*0.5
        if lr < 2e-6:
            return 2e-6
        else:
            return lr
####################################################################################################################
train_model,entity_model = build_model_from_config(config_path, checkpoint_path, seq_len=180)
train_data,dev_data = split_data(_train_data)
train_D = data_generator(train_data,32)
reduce_lr = LearningRateScheduler(scheduler, verbose=1)
best_f1 = 0
for i in range(1,2):
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=1,
                              callbacks=[reduce_lr]
                              )
    # if (i) % 2 == 0 : #两次对dev进行一次测评,并对dev结果进行保存
    print('进入到这里了哟~')
    P, R, F = predict_test_batch('dev')
    if F > best_f1 :
        train_model.save_weights(weight_name)
        print('当前第{}个epoch，准确度为{},召回为{},f1为：{}'.format(i,P,R,F))
predict_test_batch('test')