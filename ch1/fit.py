#%%
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
import matplotlib.pyplot as plt


#%% [markdown]
# 在训练数据集和测试数据集中，给定样本特征 x ，我们使用如下的三阶多项式函数来生成该样本的标签：
# $$
# y=1.2x−3.4x^2+5.6x^3+5+ϵ,
# $$
# 其中噪声项 ϵ 服从均值为0、标准差为0.1的正态分布。训练数据集和测试数据集的样本数都设为100。

#%%
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2),
                          nd.power(features, 3))
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += nd.random.normal(scale=0.1, shape=labels.shape)

#%%
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

#%%
num_epochs, loss = 100, gluon.loss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels, wd=0):
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gluon.data.DataLoader(gluon.data.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',
                            {'learning_rate': 0.01, 'wd': wd})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd',
                            {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with ag.record():
                l = loss(net(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(),
          '\nbias:', net[0].bias.data().asnumpy())

#%%
# 模型合适，正常拟合
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])
#%%
# 模型过于简单,欠拟合
fit_and_plot(poly_features[:n_train, :2], poly_features[n_train:, :2],
             labels[:n_train], labels[n_train:])
#fit_and_plot(features[:n_train, :], features[n_train:, :],
#             labels[:n_train], labels[n_train:])
#%%
# 模型合适，数据太少,过拟合
num = 5
fit_and_plot(poly_features[0:num, :], poly_features[n_train:, :],
             labels[0:num], labels[n_train:])
#%%
# 模型合适，数据太少,过拟合,加惩罚项对抗过拟合
#num = 5
fit_and_plot(poly_features[0:num, :], poly_features[n_train:, :],
             labels[0:num], labels[n_train:], 2)
