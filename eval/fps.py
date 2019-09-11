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
from baseline.encnet import EncNet
from baseline.danet import DANet
from baseline.pspnet import PSPNet
from baseline.unet import UNet
from baseline.segnet import SegNet
from model.aunetv19 import AUNet
from baseline.icnet import ICNet
from baseline.denseaspp import DenseASPP
weight_name = "denseaspp"
fine_size = 202
pad_left = 27
pad_right = 27
batch_size = 16
epoch = 200
snapshot = 4
cuda = True
save_weight = '../train_baseline/weights_split/'
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

#EncNet 10.250062183  78
#DaNet 9.871497159    81
#PSPNet 9.591554787   83
#SegNet 6.98  114
#DenseASPP 6.750459675 118
#UNet 6.527634473  122
#DSSNet 5.9 135
#ICNet 5.194065398 154


if __name__ == '__main__':
    for i in [0]:
        # Load test data
        image_test, _ = trainImageFetch(test_id[i])
        overall_pred = np.zeros((len(test_id[i]), 202, 202), dtype=np.float32)
        print(weight_name)
        # Get model
        salt = DenseASPP(1)
      #  salt = UNet()
        #salt = SegNet(3,1)
        salt = salt.to(device)
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

        e1 = cv2.getTickCount()
        for images in tqdm(test_loader, total=len(test_loader)):
            images = images.to(device)
            with torch.set_grad_enabled(False):

                # your code execution
                pred = salt(images)[0]
                #pred = salt(images)
        e2 = cv2.getTickCount()
        # 计算得到时钟时间，单位是秒
        time = (e2 - e1) / cv2.getTickFrequency()
        print(time)