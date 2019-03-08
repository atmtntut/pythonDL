import utils
from mxnet import nd
from mxnet import autograd as ag
from mxnet import gluon

#loaddata
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

#set net
net = gluon.nn.Sequential()
if False:
    net.add(gluon.nn.Dense(10))
else:
    with net.name_scope():
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(10))
net.initialize()

#train
loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.1})
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = 0., 0.
    for data, label in train_data:
        with ag.record():
            out = net(data)
            loss = loss_func(out, label)
        loss.backward()
        trainer.step(batch_size)

        label = label.astype('float32')
        train_loss += nd.mean(loss).asscalar()
        train_acc += nd.mean(out.argmax(axis=1) == label).asscalar()

    test_acc = utils.evaluate_accuracy(test_data, net)
    print('epoch %d. Loss: %f, Train acc %f, Test acc %f' % 
            (epoch, train_loss / len(train_data), train_acc / len(train_data), test_acc))
