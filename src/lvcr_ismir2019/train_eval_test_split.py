import numpy as np
from . import get_resource_path

with open(get_resource_path('data', 'all_1217.csv'), 'r') as f:
    lines=[line.strip() for line in f.readlines()]
names={line:i for i,line in enumerate(lines)}
total_songs=len(names)
TRAIN_IDS=[]
VAL_IDS=[]
TEST_IDS=[]
TEST_FOLD_LOOKUP_TABLE={}
np.random.seed(20190326)
for fold in range(5):
    with open(get_resource_path('data', f'train{fold:02d}.csv'), 'r') as f:
        result=[names[line.strip()] for line in f.readlines()]
    result_length=len(result)
    val_set_count=result_length//4
    perm=np.random.permutation(result_length)
    result=[result[i] for i in perm]
    TRAIN_IDS.append(result[:-val_set_count])
    VAL_IDS.append(result[-val_set_count:])
    with open(get_resource_path('data', f'test{fold:02d}.csv'), 'r') as f:
        data=[line.strip() for line in f.readlines()]
    TEST_IDS.append([names[i] for i in data])
    for name in data:
        TEST_FOLD_LOOKUP_TABLE[name]=fold

def get_train_set_ids(fold):
    return np.array(TRAIN_IDS[fold])

def get_val_set_ids(fold):
    return np.array(VAL_IDS[fold])

def get_test_set_ids(fold):
    return np.array(TEST_IDS[fold])

def get_test_fold_by_name(entry_name):
    if(entry_name.startswith('jam/')):
        keyword=entry_name[4:]
        if(keyword in TEST_FOLD_LOOKUP_TABLE):
            return TEST_FOLD_LOOKUP_TABLE[keyword]
    return -1
