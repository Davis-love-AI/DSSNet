import os, sys
import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


base_file_path = '../dataset' + os.sep
train_file_path = base_file_path + 'train.csv'
data = pd.read_csv(train_file_path)
# train  valid 400   test 800
train_df, test_df = train_test_split(data,
                 test_size = 0.2,
                  # hack to make stratification work
                 stratify = data.coverage_class, random_state=42)

kfold=KFold(n_splits=5,shuffle=True)
i = 1

for train_index,valid_index in kfold.split(train_df):  #对Sam数据建立5折交叉验证的划分
    ids = []
    coverage = []
    coverage_class = []
    rle_mask = []

    for index in train_index:
        ids.append(train_df.iloc[index]["id"])
        coverage.append(train_df.iloc[index]["coverage"])
        coverage_class.append(train_df.iloc[index]["coverage_class"])
        rle_mask.append(train_df.iloc[index]["rle_mask"])
    trainframe = pd.DataFrame({'id': ids, 'coverage':coverage, 'coverage_class':coverage_class, 'rle_mask': rle_mask})
    trainframe.to_csv("train"+str(i)+".csv", index=False, sep=',')
    ids.clear()
    coverage.clear()
    coverage_class.clear()
    rle_mask.clear()

    for index in valid_index:
        ids.append(train_df.iloc[index]["id"])
        coverage.append(train_df.iloc[index]["coverage"])
        coverage_class.append(train_df.iloc[index]["coverage_class"])
        rle_mask.append(train_df.iloc[index]["rle_mask"])
    testframe = pd.DataFrame({'id': ids, 'coverage':coverage, 'coverage_class':coverage_class, 'rle_mask': rle_mask})
    testframe.to_csv("valid" + str(i) + ".csv", index=False, sep=',')
    ids.clear()
    coverage.clear()
    coverage_class.clear()
    rle_mask.clear()
    i = i + 1

test_df.to_csv("test.csv", index=False, sep=',')