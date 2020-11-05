
# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 09/04/2020 下午4:07 
@Author: xinzhi yao 
Refrence Link: https://morvanzhou.github.io/tutorials/machine-learning/torch/3-02-classification/
"""

import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这
from torch.utils.data import Dataset, DataLoader


data_size = 1000
feature_size = 5

batch_size = 100
learning_rate = 0.02

hidden_size = 10
label_szie = 2

# training data
n_data = torch.ones(data_size, feature_size)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(data_size)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(data_size)



class CLS_Dataset(Dataset):
    def __init__(self, data_x_p, data_y_p, data_x_n, data_y_n):
        self.data_x = torch.cat((data_x_p, data_x_n), 0).type(torch.FloatTensor)
        self.data_y = torch.cat((data_y_p, data_y_n), ).type(torch.LongTensor)
        print('data size: {0}, positive data: {1}, negative data: {2}.'
              .format(self.data_x.size(0), len(data_y_p), len(data_y_n)))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index: int):
        return self.data_x[index], self.data_y[index]

cls_dataset = CLS_Dataset(x0, y0, x1, y1)
cls_dataloader = DataLoader(dataset=cls_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

for batch_num, (batch_data, batch_label) in enumerate(cls_dataloader):
    print(batch_data.shape, batch_label.shape)


# Visualization
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
            c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


class CLS_Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(CLS_Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值) batch_size, hidden_size
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算 batch_size, label_size
        return x


net = CLS_Net(n_feature=feature_size, n_hidden=hidden_size, n_output=label_szie)

print(net)  # net 的结构

"""
Net(
  (hidden): Linear(in_features=5, out_features=10, bias=True)
  (out): Linear(in_features=10, out_features=2, bias=True)
)
"""


optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()

report_batch_num = int((data_size/batch_size)/5)

for epoch in range(1):
    for batch_num, (batch_data, batch_label) in enumerate(cls_dataloader):
        out = net(batch_data)
        loss = loss_func(out, batch_label)
        optimizer.zero_grad()
        loss.backward()
        pred_y = torch.max(F.softmax(out, dim=-1), 1)[1].data.numpy().squeeze()
        target_y = batch_label.data.numpy()
        accuracy = sum(pred_y == target_y) / batch_size
        print('Training data: Loss：{0:.4f} Accuracy：{1:.2f}'.format(loss, accuracy))
        #if batch_num % report_batch_num == 0:
            #model_eval(net, x_testing, y_testing)
        optimizer.step()

"""
Training data: Loss：0.5926 Accuracy：0.62
Testing data: precision: 0.7632, recall: 0.5500, accuracy: 0.5500, f1: 0.6393.
Training data: Loss：0.5776 Accuracy：0.67
Training data: Loss：0.5554 Accuracy：0.69
Testing data: precision: 0.7750, recall: 0.5910, accuracy: 0.5910, f1: 0.6706.
Training data: Loss：0.5392 Accuracy：0.77
Training data: Loss：0.5319 Accuracy：0.83
Testing data: precision: 0.7910, recall: 0.6410, accuracy: 0.6410, f1: 0.7082.
Training data: Loss：0.4636 Accuracy：0.94
Training data: Loss：0.4637 Accuracy：0.95
Testing data: precision: 0.8131, recall: 0.7015, accuracy: 0.7015, f1: 0.7532.
Training data: Loss：0.4698 Accuracy：0.95
Training data: Loss：0.4411 Accuracy：0.99
Testing data: precision: 0.8367, recall: 0.7575, accuracy: 0.7575, f1: 0.7951.
Training data: Loss：0.4281 Accuracy：1.00
Training data: Loss：0.3826 Accuracy：0.99
Testing data: precision: 0.8579, recall: 0.8015, accuracy: 0.8015, f1: 0.8287.
Training data: Loss：0.3646 Accuracy：1.00
Training data: Loss：0.3899 Accuracy：0.99
Testing data: precision: 0.8712, recall: 0.8265, accuracy: 0.8265, f1: 0.8483.
Training data: Loss：0.3542 Accuracy：1.00
Training data: Loss：0.3630 Accuracy：1.00
Testing data: precision: 0.8820, recall: 0.8455, accuracy: 0.8455, f1: 0.8634.
Training data: Loss：0.3207 Accuracy：1.00
Training data: Loss：0.3445 Accuracy：1.00
Testing data: precision: 0.8959, recall: 0.8685, accuracy: 0.8685, f1: 0.8820.
Training data: Loss：0.3057 Accuracy：1.00
Training data: Loss：0.2823 Accuracy：1.00
Testing data: precision: 0.9039, recall: 0.8810, accuracy: 0.8810, f1: 0.8923.
Training data: Loss：0.2774 Accuracy：1.00
"""
