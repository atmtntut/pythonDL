import numpy as np
import matplotlib.pylab as plt
from keras.datasets import mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def step_func(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def draw():
    x = np.arange(-5.0, 5.0, 0.1)
    ys = []
    ys.append(step_func(x))
    ys.append(sigmoid(x))
    ys.append(relu(x))
    for y in ys:
        plt.plot(x, y)
    #plt.ylim(-0.1, 1.1)
    plt.show()

def init_network():
    net = {}
    net['W1'] = np.array([ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ])
    net['b1'] = np.array([0.1, 0.2, 0.3])
    net['W2'] = np.array([ [0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    net['b2'] = np.array([0.1, 0.2])
    net['W3'] = np.array([ [0.1, 0.3], [0.2, 0.4]])
    net['b3'] = np.array([0.1, 0.2])
    return net

def forward(W, b, x):
    a = np.dot(x, W) + b
    z = sigmoid(a)
    return z

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a -c )
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

if __name__=='__main__':
    #net = init_network()
    #X = np.array([1.0, 0.5])
    #z = forward(net['W1'], net['b1'], X)
    #z = forward(net['W2'], net['b2'], z)
    #y = np.dot(z, net['W3']) + net['b3']
    #print(softmax(y))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img = x_train[0]
    label = y_train[0]
    print(label)
    img_show(img)

