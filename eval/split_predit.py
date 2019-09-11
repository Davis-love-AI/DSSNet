import sys
sys.path.append('../')
from tqdm import tqdm
import os
import argparse
import cv2
import numpy as np
import pandas as pd

from data_source.data_source1 import DataSource1, trainImageFetch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.aunetv21 import AUNet

weight_name = "train_ds"
fine_size = 202
pad_left = 27
pad_right = 27
batch_size = 18
epoch = 200
snapshot = 4
cuda = True
save_weight = '../train_split/weights_split/'
max_lr = 0.01
min_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
save_pred = 'predict/'
device = torch.device('cuda' if cuda else 'cpu')

test_id = []
for i in range(5):
    ids = pd.read_csv('../dataset/data_split/test'+str(i)+'.csv')['id'].values
    test_id.append(ids)
if __name__ == '__main__':
    for i in range(1):
        # Load test data
        image_test, _ = trainImageFetch(test_id[i])

        overall_pred = np.zeros((len(test_id[i]), 128, 128), dtype=np.float32)

        # Get model
        salt = AUNet()
        salt = salt.to(device)
        pred_null = []
        pred_flip = []
         # Load weight
        param = torch.load(save_weight + weight_name +str(i)+ '.pth')
        salt.load_state_dict(param)
        # Create DataLoader
        test_data = DataSource1(image_test, mode='test', fine_size=fine_size, pad_left=pad_left, pad_right=pad_right)
        test_loader = DataLoader(
                                test_data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=8,
                                pin_memory=True)
        # Prediction with no TTA test data
        salt.eval()
        for images in tqdm(test_loader, total=len(test_loader)):
            images = images.to(device)
            with torch.set_grad_enabled(False):
               # pred, _, _, _, _ = salt(images)
                pred = salt(images)
                pred = F.sigmoid(pred).squeeze(1).cpu().numpy()
            pred = pred[:, 0:128, 0:128]
            pred_null.append(pred)

            # Prediction with horizontal flip TTA test data
        test_data = DataSource1(image_test, mode='test', is_tta=True, fine_size=fine_size, pad_left=pad_left,
                                    pad_right=pad_right)
        test_loader = DataLoader(test_data,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=8,
                                pin_memory=True)

        salt.eval()
        for images in tqdm(test_loader, total=len(test_loader)):
            images = images.to(device)
            with torch.set_grad_enabled(False):
                #pred, _, _, _, _ = salt(images)
                pred = salt(images)
                pred = F.sigmoid(pred).squeeze(1).cpu().numpy()
                pred = pred[:, 0:128, 0:128]
                for idx in range(len(pred)):
                    pred[idx] = cv2.flip(pred[idx], 1)
                pred_flip.append(pred)
        pred_null = np.concatenate(pred_null).reshape(-1, 128, 128)
        pred_flip = np.concatenate(pred_flip).reshape(-1, 128, 128)
        overall_pred += (pred_null + pred_flip) / 2


        # Save prediction
        if fine_size != 101:
            overall_pred_101 = np.zeros((len(test_id[i]), 101, 101), dtype=np.float32)
            for idx in range(len(test_id[i])):
                overall_pred_101[idx] = cv2.resize(overall_pred[idx], dsize=(101, 101))
            np.save(save_pred +weight_name+  str(i), overall_pred_101)
        else:
            np.save(save_pred+weight_name+  str(i), overall_pred)
