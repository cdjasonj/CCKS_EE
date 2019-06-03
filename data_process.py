#数据后处理，把数据类型为其他的拼接成NAN
import json
import numpy

result_train = 'output/result_A.txt'
result_no_train = 'inputs/test_data_me_no_train.json'

result_A = json.load(open(result_train,encoding='utf-8'))
result_B = json.load(open(result_no_train,encoding='utf-8'))

result = []
for result in result_B:
    result.append(str(result[id]+','+'NaN'))

result+=result_A