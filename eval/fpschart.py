# 导入必要的模块
import numpy as np
import matplotlib.pyplot as plt
##EncNet 10.250062183  78
#DaNet 9.871497159    81
#PSPNet 9.591554787   83
#SegNet 6.98  114
#DenseASPP 6.750459675 118
#UNet 6.527634473  122
#DSSNet 5.9 135
#ICNet 5.194065398 154


ious = [82.1, 81.9, 86.0, 87.3, 86.4, 78.8, 85.6, 88.8]
fps = [122, 114, 154, 83, 81, 118, 78, 135]
# 产生测试数据

fig = plt.figure()
ax1 = fig.add_subplot(111)
c = ['b','c','g','k','m','#9999ff','y','r']
mark = ['o', 'v','^','<','>','s','D','*']
# 设置标题
# 设置X轴标签
plt.xlabel('FPS')
# 设置Y轴标签
plt.ylabel('Accuracy(IoU%)')
# 画散点图
for i in range(8):
    ax1.scatter(fps[i], ious[i], c=c[i],s=100, marker=mark[i])

# ax1.scatter(x[0], y[0], c=c[0], marker='v')
# ax1.scatter(x[1], y[1], c=c[1], marker='o')
# 设置图标
#plt.legend(['U-Net','SegNet', 'ICNet','PSPNet','DANet','DenseASPP', 'EncNet','DSSNet'])
# 显示所画的图
plt.show()