import json
import numpy as np


train_data = json.load(open('inputs/train_data_me.json',encoding='utf-8'))
test_data = json.load(open('inputs/dev_data_me.json',encoding='utf-8'))
test_data_no_train = json.load(open('inputs/test_data_me_no_train.json',encoding='utf-8'))

train_dic = {}
for data in train_data:
    if data['entity'] not in train_dic:
        train_dic[data['entity']] = []
        train_dic[data['entity']].append(data['event_type'])
    else:
        train_dic[data['entity']].append(data['event_type'])


test_dic_ds = []
entities = list(train_dic.keys())
for data in test_data:
    temp = []
    flag = 0
    for entity in entities:
        if str(entity) in data['text']:
            if data['event_type'] in train_dic[entity]:
                temp.append(str(entity))
                if len(temp) == 1: #只保留单实体得
                    dic = {}
                    dic['id'] = data['id']
                    dic['entity'] = str(entity)
                    flag = 1
                    test_dic_ds.append(dic)

all_ids = []
relu_ids = []
for data in test_data:
    all_ids.append(data['id'])
for data in test_dic_ds:
    relu_ids.append(data['id'])
other_ids = list(set(all_ids) - set(relu_ids))

with open('output/relu_ds.txt','w',encoding='utf-8') as fr:
    for data in test_dic_ds:
        id = data['id']
        entity = data['entity']
        fr.write(str(id) + ',' + str(entity) + '\n')
    for id in other_ids:
        entity = ' '
        fr.write(str(id) + ',' + str(entity) + '\n')

    entity = ' '
    fr.write(str(213155) + ',' + str(entity) + '\n')

result_a = []
with open('output/result_A.txt',encoding='utf-8') as fr:
    for line in fr:
        dic = {}
        dic['id'] = line.split(',')[0].strip()
        dic['entity'] = line.split(',')[1].strip()
        result_a.append(dic)


new_data = []
count = 0
for data in result_a:
    flag = 0
    if data['entity'] :
        flag = 1
        new_data.append(data)
    else:
        id = data['id']
        for relu_data in test_dic_ds:
            if int(relu_data['id']) == int(id):
                new_data.append(relu_data)
                count+=1
                flag = 1
    if flag == 0 :
        new_data.append(data)
print('一共填补了{}个'.format(count))

count_ = 0
for data in new_data:
    if data['entity']:
        id = int(data['id'])
        for relu_data in test_dic_ds:
            if int(relu_data['id']) == id:
                if  data['entity'] in relu_data['entity'] and len(data['entity']) < len(relu_data['entity']):
                    print(data['entity'])
                    print(relu_data['entity'])
                    data['entity'] = relu_data['entity']
                    print(' ')
                    count_+=1

print('一共修复了{}条'.format(count_))
with open('output/combine_data.txt','w',encoding='utf-8') as fr:
    for data in new_data:
        id = data['id']
        entity = data['entity']
        fr.write(str(id) + ',' + str(entity) + '\n')