import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from data_source.data_source_split import testMaskFetch
import torch.nn.functional as F
from tqdm import tqdm
from utils.eval_method import iou
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', default='', type=str, help='Model version')
args = parser.parse_args()

def iou_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return metric

if __name__ == '__main__':
    save_predict = "predict/"
    model_name = "danet"
    c_p = {i:[] for i in range(1, 11)}
    c_t = {i:[] for i in range(1, 11)}
    c_ids = {i: [] for i in range(1, 11)}
    all_predicts = []
    all_truths = []
    for i in range(5):
        test_df = pd.read_csv('../dataset/data_split/test' + str(i) + '.csv')
        truths = testMaskFetch(test_df["id"].values)
        predicts = np.load(save_predict + model_name + str(i) + ".npy")
        all_predicts.append(predicts)
        all_truths.append(truths)
        for index, row in tqdm(test_df.iterrows()):
            c = row["coverage_class"]
            if (c == 0 or c == 1):
                c_p[1].append(predicts[index].reshape(1, 101, 101))
                c_t[1].append(truths[index].reshape(1, 101, 101))
                c_ids[1].append(row["id"])
            for i in range(2,11):
                if (c == i):
                    c_p[i].append(predicts[index].reshape(1, 101, 101))
                    c_t[i].append(truths[index].reshape(1, 101, 101))
                    c_ids[i].append(row["id"])
    for i in range(1, 11):
        p = np.concatenate(c_p[i])
        t = np.concatenate(c_t[i])

        np.save(model_name + str(i), iou_batch(t, np.int32(p > 0.5)))
        # np.save("class_" + str(i),c_ids[i])




