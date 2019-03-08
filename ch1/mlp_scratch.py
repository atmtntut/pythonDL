import utils
from mxnet import nd
from mxnet import autograd as ag
from mxnet import gluon

#loaddata
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

#set net
#net = gluon.nn.Sequential()
#if False:
#    net.add(gluon.nn.Dense(10))
#else:
#    with net.name_scope():
#        net.add(gluon.nn.Flatten())
#        net.add(gluon.nn.Dense(10))
#net.initialize()
num_hiddens = 512
num_hiddens1 = 128
W1 = nd.random.normal(scale=0.1, shape=(784, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.1, shape=(num_hiddens, num_hiddens1))
b2 = nd.zeros(num_hiddens1)
W3 = nd.random.normal(scale=0.1, shape=(num_hiddens1, 10))
b3 = nd.zeros(10)
params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, 0)

def net(X):
    X = X.reshape((-1, 784))
    H = relu(nd.dot(X, W1) + b1)
    H1 = relu(nd.dot(H, W2) + b2)
    return nd.dot(H1, W3) + b3

#train
loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
lr = 0.03
#trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = 0., 0.
    for data, label in train_data:
        with ag.record():
            out = net(data)
            loss = loss_func(out, label)
        loss.backward()
        #trainer.step(batch_size)
        utils.sgd(params, lr, batch_size)

        label = label.astype('float32')
        train_loss += nd.mean(loss).asscalar()
        train_acc += nd.mean(out.argmax(axis=1) == label).asscalar()

    test_acc = utils.evaluate_accuracy(test_data, net)
    print('epoch %d. Loss: %f, Train acc %f, Test acc %f' % 
            (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
