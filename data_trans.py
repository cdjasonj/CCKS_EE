import json
import numpy as np
import pandas as pd
import codecs

train_data = pd.read_csv('inputs/event_type_entity_extract_train.csv',header=None)
dev_data = pd.read_csv('inputs/event_type_entity_extract_eval.csv',header=None)
train_data_me = []
dev_data_me = []

def train_trans_data(item):
    dic = {}
    dic['id'] = item[0]
    dic['text'] = item[1]
    dic['event_type'] = item[2]
    dic['entity'] = item[3]
    train_data_me.append(dic)

def dev_trans_data(item):
    dic = {}
    dic['id'] = item[0]
    dic['text'] = item[1]
    dic['event_type'] = item[2]
    dev_data_me.append(dic)

a = train_data.apply(train_trans_data,axis=1)
b = dev_data.apply(dev_trans_data,axis=1)

with codecs.open('inputs/train_data_me.json', 'w', encoding='utf-8') as f:
    json.dump(train_data_me, f, indent=4, ensure_ascii=False)
with codecs.open('inputs/test_data_me.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data_me, f, indent=4, ensure_ascii=False)

chars = {}
min_count=2
texts = [data['text'] for data in train_data_me]
texts += [data['text'] for data in dev_data_me]

for text in texts:
    for c in text:
        chars[c] = chars.get(c,0)+1
id2char = {i+2:j for i,j in enumerate(chars)} #padding:0,UNK1
char2id = {j:i for i,j in id2char.items()}
with codecs.open('inputs/all_chars_me.json','w',encoding='utf-8') as f:
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

#event_type2id
event_types = list(set([data['event_type'] for data in train_data_me]))

event_types += [data['event_type'] for data in dev_data_me]
event_types = list(set(event_types))
event2id = {i:j for i,j in enumerate(event_types)}

with codecs.open('inputs/event2id.json','w',encoding='utf-8') as f:
    json.dump(event2id,f,indent=4,ensure_ascii=False)