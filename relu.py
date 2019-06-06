import json
import numpy as np
import operator

train_data = json.load(open('inputs/train_data_me.json',encoding='utf-8'))
test_data = json.load(open('inputs/dev_data_me.json',encoding='utf-8'))
test_data_no_train = json.load(open('inputs/test_data_me_no_train.json',encoding='utf-8'))
test_dic_ds = []
with open('output/relu_ds.txt','r',encoding='utf-8') as fr:
    for line in fr:
        dic = {}
        id = line.split(',')[0].strip()
        entity = line.split(',')[1].strip()
        dic['id'] = id
        dic['entity'] = entity
        test_dic_ds.append(dic)

def process_data(result_a):
    # train_dic = {}
    # for data in train_data:
    #     if data['entity'] not in train_dic:
    #         train_dic[data['entity']] = []
    #         train_dic[data['entity']].append(data['event_type'])
    #     else:
    #         train_dic[data['entity']].append(data['event_type'])


    # test_dic_ds = []
    # entities = list(train_dic.keys())
    # for data in test_data:
    #     temp = []
    #     flag = 0
    #     for entity in entities:
    #         if str(entity) in data['text']:
    #             if data['event_type'] in train_dic[entity]:
    #                 temp.append(str(entity))
    #                 if len(temp) == 1: #只保留单实体得
    #                     dic = {}
    #                     dic['id'] = data['id']
    #                     dic['entity'] = str(entity)
    #                     flag = 1
    #                     test_dic_ds.append(dic)

    # all_ids = []
    # relu_ids = []
    # for data in test_data:
    #     all_ids.append(data['id'])
    # for data in test_dic_ds:
    #     relu_ids.append(data['id'])
    #
    # other_ids = list(set(all_ids) - set(relu_ids))

    # with open('output/relu_ds.txt','w',encoding='utf-8') as fr:
    #     for data in test_dic_ds:
    #         id = data['id']
    #         entity = data['entity']
    #         fr.write(str(id) + ',' + str(entity) + '\n')
    #     for id in other_ids:
    #         entity = ' '
    #         fr.write(str(id) + ',' + str(entity) + '\n')
    #
    #     entity = ' '
    #     fr.write(str(213155) + ',' + str(entity) + '\n')


    new_data = []
    count = 0
    for data in result_a:
        flag = 0
        if data['entity']:
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

    #这里有bug，等会儿再说
    count_ = 0
    for data in new_data:
        if data['entity']:
            id = int(data['id'])
            for relu_data in test_dic_ds:
                if int(relu_data['id']) == id:
                    if  (data['entity'] in relu_data['entity']) and (len(data['entity']) < len(relu_data['entity'])):
                        # print(data['entity'])
                        # print(relu_data['entity'])
                        data['entity'] = relu_data['entity']
                        # print(' ')
                        count_+=1
    print('一共修复了{}条'.format(count_))

    # with open('output/combine_data.txt','w',encoding='utf-8') as fr:
    #     for data in new_data:
    #         id = data['id']
    #         entity = data['entity']
    #         fr.write(str(id) + ',' + str(entity) + '\n')
    combine_data = {}
    for data in new_data:
        id = data['id']
        entity = data['entity']
        dic['id'] = id
        dic['entity'] = entity
        #坑jb爹，id有个重复的
        combine_data[id] = entity
    sorted_combine_data = sorted(combine_data.items(),key=lambda  item:item[0])

    return sorted_combine_data


def leak_data(result_a):

    train_text = {}
    for data in train_data:
        if data['text'] not in train_text:
            train_text[data['text']] = []
            train_text[data['text']].append((data['id'],data['entity']))
        else:
            train_text[data['text']].append((data['id'],data['entity']))

    # 找出相同文本只有一个实体的样本
    singe_entity_text = {}
    for text, entities in train_text.items():
        if len(entities) == 1:
            singe_entity_text[text] = entities[0]

    #保存数据泄露字典
    leak_text = {}
    for data in test_data:
        if data['text'] in singe_entity_text:
            leak_text[data['text']] = singe_entity_text[data['text']]

    leak_data = list(leak_text.values())
    count = 0
    for id,entity in leak_data:
            for _result in result_a:
                if _result['id'] == id and _result['entity'] != entity:
                    _result['entity'] = entity
                    count+=1
    print('泄露数据修改{}条'.format(count))


