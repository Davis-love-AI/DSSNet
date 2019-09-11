import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from data_source.data_source_split import testMaskFetch
import torch.nn.functional as F
from tqdm import tqdm
from utils.eval_method import iou_batch,kaggle_iou_batch
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', default='', type=str, help='Model version')
args = parser.parse_args()
if __name__ == '__main__':
    save_predict = "predict/"
    model_name = args.model_name
    c_p = {i:[] for i in range(1, 11)}
    c_t = {i:[] for i in range(1, 11)}

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
            for i in range(2,11):
                if (c == i):
                    c_p[i].append(predicts[index].reshape(1, 101, 101))
                    c_t[i].append(truths[index].reshape(1, 101, 101))
    all_predicts = np.concatenate(all_predicts)
    all_truths = np.concatenate(all_truths)
    all_map = kaggle_iou_batch(all_truths, np.int32(all_predicts > 0.5))
    all_iou = iou_batch(all_truths , np.int32(all_predicts > 0.5))
    print("all_count:", all_predicts.shape[0])
    print("all_map:", all_map)
    print("all_iou:", all_iou)

    for i in range(1, 11):
        p = np.concatenate(c_p[i])
        t = np.concatenate(c_t[i])
        c_map = kaggle_iou_batch(t, np.int32(p > 0.5))
        c_iou = iou_batch(t, np.int32(p > 0.5))
        print(i,"count:",p.shape[0])
        print(i,"map:",c_map)
        print(i, "iou:", c_iou)




