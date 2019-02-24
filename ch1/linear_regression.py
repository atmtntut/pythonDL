from matplotlib import pyplot as plt
from mxnet import nd
from mxnet import autograd as ag
import random

# 生成数据
num_inputs = 2
num_examples = 1000
tw = [2, -3.4]
tb = 4.2
xdata = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
ydata = tw[0]*xdata[:, 0] + tw[1]*xdata[:, 1] + tb
ydata += nd.random.normal(scale=0.01, shape=ydata.shape)

# 迭代取一批数据
def data_iter(batch_size, X, Y):
    num_examples = len(X)
    ids = list(range(num_examples))
    random.shuffle(ids)
    for i in range(0, num_examples, batch_size):
        j = nd.array(ids[i:min(i+batch_size, num_examples)])
        yield X.take(j), Y.take(j)

# 初始化w,b
w = nd.random.normal(scale=0.1, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()

# 定义神经网络,这只是个单层感知机
def linreg(X, w, b):
    return nd.dot(X, w) + b

# 残差
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

# 随机梯度下降
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr*param.grad/batch_size

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

epochs = []
losses = []
# 训练
for epoch in range(num_epochs):
    for x,y in data_iter(batch_size, xdata, ydata):
        with ag.record():
            l = loss(net(x, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(x, w, b), y)

    epochs.append(epoch)
    print(train_l.mean().asscalar())
    losses.append(train_l.mean().asscalar())
    #print(f'epoch:{epoch}, loss:{train_l.mean().asnumpy()}')
plt.plot(epochs, losses)
plt.show()
print(w, tw)
print(b, tb)
