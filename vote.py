import json
import numpy as np
import os
from relu import process_data
# result_path = 'output/bagging_result_test_{}.txt'
result_path = 'output/vote.txt'

result = []
for i in range(1):
    temp = []
    path = result_path
    with open(path,'r',encoding='utf-8') as fr:
        for line in fr:
            dic = {}
            id = line.split(',')[0].strip()
            entity = line.split(',')[1].strip()
            dic['id'] = id
            dic['entity'] = entity
            temp.append(dic)
    result.append(temp)
    # result.append(process_data(temp))
def vote_result():
    #取最高的票数的entity
    voted_data = []
    id_dic = {} #按照id来保存字典 id : {entity:num}
    for _result in result:
        # for id,entity in _result:
        for data in _result:
            id = data['id']
            entity = data['entity']
            if id in id_dic:
                 id_dic[id][entity] = id_dic[id].get(entity,0)+1
            else:
                id_dic[id] = {}
                id_dic[id][entity] = id_dic[id].get(entity, 0) + 1
    #将id_dic[id]按照value进行排序
    for id,dic in id_dic.items():
        temp_dic = {}
        temp_dic['id'] = id
        # #取出现次数最多的enitty
        # print(list(dic.keys()))
        # print(id)
        _,entity = max(zip(dic.values(), dic.keys())) #这样不行，把他全部换成list,取最长的
        # print(entity)
        # entities = list(dic.keys())
        # maxlen = 0
        # entity = ''
        # for e in entities:
        #     if len(e) > maxlen:
        #         entity = e
        #         maxlen = len(e)
        # print(entities)
        # print(entity)
        # print(' ')
        temp_dic['entity'] = entity
        voted_data.append(temp_dic)
    return voted_data

voted_data = vote_result()
process_data(voted_data)

with open('output/voted_data_5.txt','w',encoding='utf-8') as fr:
    for data in voted_data:
        id = data['id']
        entity = data['entity']
        fr.write(str(id) + ',' + str(entity) + '\n')

    entity = ''
    fr.write(str(213155) + ',' + '民生浦发' + '\n')