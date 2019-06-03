import json
import numpy as np
from random import sample
import codecs
#只划分0.2作为验证
#将test_data按照是否类别是其他分成两个部分。
train_data = json.load(open('inputs/train_data_me.json',encoding='utf-8'))
test_data = json.load(open('inputs/test_data_me.json',encoding='utf-8'))

# def get_event_dic(data):
#     """
#     得到event_type的数量字典
#     :param data:
#     :return:
#     """
#     dic = {}
#     prob_dic = {}
#     for _data in data:
#         dic[_data['event_type']] = dic.get(_data['event_type'],0)+1
#     for key,value in dic.items():
#         prob_dic[key] = value/len(data)
#     return prob_dic

def split_test_data(test_data):
    """
    test_data 按照需要被预测和不需要被预测进切分
    :param test_data:
    :return:
    """
    test_data_train = []
    test_data_no_train = []
    for data in test_data:
        if data['event_type'] == '其他':
            test_data_no_train.append(data)
        else:
            test_data_train.append(data)
    return test_data_train,test_data_no_train

def clean_train_data(_train_data):
    all_data = []
    for data in _train_data:
        if data['event_type'] != '其他':
            all_data.append(data)
    return all_data

def split_dev(train_data):
    """
    按照test_data的事件类型分布来划分dev
    :param train_data:
    :param test_data:
    :return:
    """
    dev_data = []
    new_train_data = []
    event_index = {}
    for index,data in enumerate(train_data):
        if data['event_type'] not in event_index:
            event_index[data['event_type']] = []
            event_index[data['event_type']].append(index)
        else:
            event_index[data['event_type']].append(index)

    dev_event_index = {}
    train_event_index = {}
    for key,value in event_index.items():
        sample_index = sample(value,round(len(value)*0.2))
        dev_event_index[key] = sample_index

        train_index = list(set(value) - set(sample_index))
        train_event_index[key] = train_index

    for key,value in dev_event_index.items():
        for index in value:
            dev_data.append(train_data[index])

    for key,value in train_event_index.items():
        for index in value:
            new_train_data.append(train_data[index])

    np.random.shuffle(dev_data)
    np.random.shuffle(new_train_data)


    return new_train_data,dev_data

test_data_train,test_data_no_train = split_test_data(test_data)
train_data = clean_train_data(train_data)
train_data,dev_data = split_dev(train_data)

with codecs.open('inputs/dev_data_me.json','w',encoding='utf-8') as f:
    json.dump(dev_data,f,indent=4,ensure_ascii=False)

with codecs.open('inputs/train_data_me.json','w',encoding='utf-8') as f:
    json.dump(train_data,f,indent=4,ensure_ascii=False)

with codecs.open('inputs/test_data_me_train.json','w',encoding='utf-8') as f:
    json.dump(test_data_train,f,indent=4,ensure_ascii=False)

with codecs.open('inputs/test_data_me_no_train.json','w',encoding='utf-8') as f:
    json.dump(test_data_no_train,f,indent=4,ensure_ascii=False)
