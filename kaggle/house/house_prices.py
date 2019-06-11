#%%
from mxnet import gluon, init
from mxnet import ndarray as nd
from mxnet import autograd as ag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%%
train_data = pd.read_csv('kaggle/house/data/train.csv')
test_data = pd.read_csv('kaggle/house/data/test.csv')
#train_data = pd.read_csv('./data/train.csv')
#test_data  = pd.read_csv('./data/test.csv')
print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4, [0,1,2,3,-2,-1]])
print(test_data.iloc[0:4, [0,1,2,3,-2,-1]])

#%%
# 数据预处理
# 将train,test合并到一起
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 筛选出数字的特征
num_featrues = all_features.dtypes[all_features.dtypes != 'object'].index
print(num_featrues)
# 特征值标准化
all_features[num_featrues] = all_features[num_featrues].apply(
    lambda x: (x - x.mean()) / (x.std())
)
# 缺失项补零,标准化后均值为0,其实是填均值
all_features[num_featrues] = all_features[num_featrues].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

#%%
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1,1))

#%%
loss = gluon.loss.L2Loss()
def get_net():
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(100, activation='relu'))
    net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

#%%
def log_rmse(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with ag.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

#%%
k, num_epochs, lr, weight_decay, batch_size = 3, 50, 0.03, 0.3, 50
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
      % (k, train_l, valid_l))