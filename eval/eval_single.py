import numpy as np
import pandas as pd
from data_source.data_source_split import testMaskFetch
import torch.nn.functional as F
from utils.metric import do_kaggle_metric,do_iou
from tqdm import tqdm
from utils.eval_method import iou_batch,kaggle_iou_batch



if __name__ == '__main__':
    save_predict = "predict/"
    model_name = "train_ds"
    i = 0
    test_df = pd.read_csv('../dataset/data_split/test' + str(i) + '.csv')
    truths = testMaskFetch(test_df["id"].values).reshape(800,1,101,101)
    predicts = np.load(save_predict + model_name + str(i) + ".npy")
    predicts = predicts.reshape(800,1,101,101)
    score = kaggle_iou_batch(truths, np.int32(predicts > 0.5))
    iou = iou_batch(truths , predicts > 0.5)
    # iou = iou.mean()
    print("score:" + str(score))
    print("iou:"+ str(iou))