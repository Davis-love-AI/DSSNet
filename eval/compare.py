import numpy as np
count = 5

for i in [10]:
    ids = np.load("iou/class_"+str(i)+".npy")
    dssnet_iou = np.load("iou/dssnet/dssnet"+str(i)+".npy")
    pspnet_iou = np.load("iou/danet/danet" + str(i) + ".npy")
    for j in range(ids.shape[0]):
        cha = dssnet_iou[j] - pspnet_iou[j]
        if(cha > 0.0 and cha < 0.1 and dssnet_iou[j] > 0.8):
            print(ids[j] + " cha"+ str(cha))
