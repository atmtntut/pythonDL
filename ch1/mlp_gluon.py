import utils
from mxnet import nd
from mxnet import autograd as ag
from mxnet import gluon
from mxnet import init

#loaddata
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

#set net
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(512, activation='relu'))
net.add(gluon.nn.Dense(256, activation='relu'))
net.add(gluon.nn.Dense(10))
net.initialize(init.Normal(sigma=0.1))

#train
loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
lr = 0.03
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = 0., 0.
    for data, label in train_data:
        with ag.record():
            out = net(data)
            loss = loss_func(out, label)
        loss.backward()
        trainer.step(batch_size)
        #utils.sgd(params, lr, batch_size)

        label = label.astype('float32')
        train_loss += nd.mean(loss).asscalar()
        train_acc += nd.mean(out.argmax(axis=1) == label).asscalar()

    test_acc = utils.evaluate_accuracy(test_data, net)
    print('epoch %d. Loss: %f, Train acc %f, Test acc %f' % 
            (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
