import sys

sys.path.append('../')
import os
import numpy as np
import pandas as pd

from utils.metric import do_kaggle_metric
from loss.lovasz_losses import lovasz_hinge
from data_source.data_source1 import DataSource1, trainImageFetch
from model.xception_encode import AUNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


model = "xception_encode"
fine_size = 202
pad_left = 27
pad_right = 27
batch_size = 18
epoch = 200
snapshot = 4
cuda = True
save_weight = 'weights_split/'
max_lr = 0.01
min_lr = 0.001
momentum = 0.9
weight_decay = 1e-4



weight_name = model

if not os.path.isdir(save_weight):
    os.mkdir(save_weight)

device = torch.device('cuda' if cuda else 'cpu')

fold_train = []
fold_valid = []
for i in range(1):
    train_ids = pd.read_csv('../dataset/data_split/train'+str(i)+'.csv')['id'].values
    fold_train.append(train_ids)
    valid_ids = pd.read_csv('../dataset/data_split/valid' + str(i) + '.csv')['id'].values
    fold_valid.append(valid_ids)


def test(test_loader, model):
    running_loss = 0.0
    predicts = []
    truths = []

    model.eval()
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.set_grad_enabled(False):
            outputs, _, _, _, _ = model(inputs)
            outputs = outputs[:, :, 0:128, 0:128].contiguous()
        loss = lovasz_hinge(outputs.squeeze(1), masks.squeeze(1))
        predicts.append(F.sigmoid(outputs).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)

        predicts.append(F.sigmoid(outputs).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)

    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()
    precision, _, _ = do_kaggle_metric(predicts, truths, 0.5)
    precision = precision.mean()
    epoch_loss = running_loss / val_data.__len__()
    return epoch_loss, precision


def train(train_loader, model):
    running_loss = 0.0
    data_size = train_data.__len__()

    model.train()
    # for inputs, masks, labels in progress_bar(train_loader, parent=mb):
    for inputs, masks, labels in train_loader:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            logit, logit1, logit2, logit3, logit4 = model(inputs)
            loss0 = lovasz_hinge(logit.squeeze(1), masks.squeeze(1))
            loss1 = lovasz_hinge(logit1.squeeze(1), masks.squeeze(1))
            loss2 = lovasz_hinge(logit2.squeeze(1), masks.squeeze(1))
            loss3 = lovasz_hinge(logit3.squeeze(1), masks.squeeze(1))
            loss4 = lovasz_hinge(logit4.squeeze(1), masks.squeeze(1))
            loss = loss0 + (0.6*loss1 + 0.2*loss2 + 0.1*loss3 + 0.1*loss4)*0.01
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # mb.child.comment = 'loss: {}'.format(loss.item())
    epoch_loss = running_loss / data_size
    return epoch_loss


if __name__ == '__main__':
    scheduler_step = epoch // snapshot
    # Get Model
    salt = AUNet()
    salt.to(device)
    for idx in range(1):

        # Setup optimizer
        optimizer = torch.optim.SGD(salt.parameters(), lr=max_lr, momentum=momentum,
                                    weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, min_lr)

        # Load data
        train_id = fold_train[idx]
        val_id = fold_valid[idx]

        X_train, y_train = trainImageFetch(train_id)
        X_val, y_val = trainImageFetch(val_id)

        train_data = DataSource1(X_train, mode='train', mask_list=y_train, fine_size=fine_size,
                                 pad_left=pad_left,
                                 pad_right=pad_right)
        train_loader = DataLoader(
            train_data,
            shuffle=RandomSampler(train_data),
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True)

        val_data = DataSource1(X_val, mode='val', mask_list=y_val, fine_size=fine_size, pad_left=pad_left,
                               pad_right=pad_left)
        val_loader = DataLoader(
            val_data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True)

        num_snapshot = 0
        best_acc = 0

        # mb = master_bar(range(args.epoch))
        # for epoch in mb:
        for epoch in range(epoch):
            train_loss = train(train_loader, salt)
            val_loss, accuracy = test(val_loader, salt)
            lr_scheduler.step()

            if accuracy > best_acc:
                best_acc = accuracy
                best_param = salt.state_dict()

            if (epoch + 1) % scheduler_step == 0:
                optimizer = torch.optim.SGD(salt.parameters(), lr=max_lr, momentum=momentum,
                                            weight_decay=weight_decay)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, min_lr)
                num_snapshot += 1

            print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch + 1, train_loss,
                                                                                              val_loss, accuracy))
        torch.save(best_param, save_weight + weight_name + str(idx) + '.pth')