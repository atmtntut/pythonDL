from matplotlib import pyplot as plt
from mxnet import nd
from mxnet import autograd as ag
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet import init
from mxnet import gluon
import random

# 生成数据
num_inputs = 2
num_examples = 1000
tw = [2, -3.4]
tb = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = tw[0]*features[:, 0] + tw[1]*features[:, 1] + tb
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 迭代取一批数据
batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# 定义神经网络,这只是个单层感知机
net = nn.Sequential()
net.add(nn.Dense(1))

# 初始化w,b
#net.initialize(init.Normal(sigma=0.1))
net.initialize()

# 残差
loss = gloss.L2Loss()

epochs = []
losses = []
# 训练
num_epochs = 5
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
for epoch in range(num_epochs):
    for x,y in data_iter:
        with ag.record():
            l = loss(net(x), y)
        l.backward()
        trainer.step(batch_size)
        #sgd([w, b], lr, batch_size)
    train_l = loss(net(x), y)

    epochs.append(epoch)
    print(train_l.mean().asscalar())
    losses.append(train_l.mean().asscalar())
    #print(f'epoch:{epoch}, loss:{train_l.mean().asnumpy()}')
plt.plot(epochs, losses)
plt.show()
print(net[0].weight.data(), tw)
print(net[0].bias.data(), tb)
