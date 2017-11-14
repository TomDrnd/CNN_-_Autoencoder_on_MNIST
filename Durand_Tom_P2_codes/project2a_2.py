from load import mnist
import numpy as np
import pylab
import pickle

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

# 2 convolution layer, 2 max pooling layer, 1 fully connected layer and a softmax layer

np.random.seed(10)
batch_size = 128
noIters = 100
learning_rate = 0.05
momentum = 0.1
decay = 1e-04

def init_weights_bias_velocity4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros(filter_shape[0], dtype=d_type)
    v_values = np.zeros(filter_shape, dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True), theano.shared(v_values,borrow=True)

def init_weights_bias_velocity2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros(filter_shape[1], dtype=d_type)
    v_values = np.zeros(filter_shape, dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True), theano.shared(v_values,borrow=True)

def model(X, w1, b1, w2, b2, w3, b3, w4, b4):
    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    pool_dim = (2, 2)
    o1 = pool.pool_2d(y1, pool_dim, ignore_border=True)
    y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
    o2 = pool.pool_2d(y2, pool_dim, ignore_border=True)
    o2_f = T.flatten(o2, outdim=2)
    y3 = T.nnet.relu(T.dot(o2_f, w3) + b3)
    pyx = T.nnet.softmax(T.dot(y3, w4) + b4)
    return y1, o1, y2, o2, y3, pyx

def sgd(cost, params, velocity):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    #updates for velocity and weights
    for v, g, p in zip(velocity, grads[0::2], params[0::2]):
        updates.append([v, momentum*v - (g + decay*p)*learning_rate])
        updates.append([p, p+momentum*v-(g + decay*p)*learning_rate])
    #updates for biases
    for g, p in zip(grads[1::2], params[1::2]):
        updates.append([p, p - (g + decay*p)*learning_rate])
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]


X = T.tensor4('X')
Y = T.matrix('Y')

num_filters_h1 = 15
num_filters_h2 = 20
w1, b1, v1 = init_weights_bias_velocity4((num_filters_h1, 1, 9, 9), X.dtype)
w2, b2, v2 = init_weights_bias_velocity4((num_filters_h2, num_filters_h1, 5, 5), X.dtype)
w3, b3, v3 = init_weights_bias_velocity2((num_filters_h2*3*3, 100), X.dtype)
w4, b4, v4 = init_weights_bias_velocity2((100, 10), X.dtype)

y1, o1, y2, o2, y3, py_x  = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]
velocity = [v1, v2, v3, v4]

updates = sgd(cost, params, velocity)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], outputs=[y1, o1, y2, o2], allow_input_downcast=True)

a = []
cost = []
for i in range(noIters):
    if i%10==0:
        print(i)
    c=0
    trX, trY = shuffle_data (trX, trY)
    teX, teY = shuffle_data (teX, teY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c += train(trX[start:end], trY[start:end])
    cost.append(c)
    a.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    print(a[i])

#This part will only work if at least the project2a_1.py has been executed

#acc = pickle.load(file('accuracy_mnist.pickle'))
#pylab.plot(range(noIters), a, '-r', label='GD with momentum')
#pylab.savefig('project2a_2_test_accuracy.png')
#pickle.dump(acc, file('accuracy_mnist.pickle', 'w'))

#acc = pickle.load(file('cost_mnist.pickle'))
#pylab.plot(range(noIters), cost, '-r', label='GD with momentum')
#pylab.savefig('project2a_2_cost.png')
#pickle.dump(acc, file('cost_mnist.pickle', 'w'))

pylab.figure()
pylab.plot(range(noIters), a, '-r', label='GD with momentum')
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.savefig('project2a_2_test_accuracy.png')

pylab.figure()
pylab.plot(range(noIters), cost, '-r', label='GD with momentum')
pylab.xlabel('epochs')
pylab.ylabel('cross-entropy')
pylab.savefig('project2a_2_cost.png')

w = w1.get_value()
pylab.figure()
pylab.gray()
for i in range(num_filters_h1):
    pylab.subplot(5, 3, i+1); pylab.axis('off'); pylab.imshow(w[i,:,:,:].reshape(9,9))
#pylab.title('filters h0 learned')
pylab.savefig('project2a_2_filters_h0.png')

w = w2.get_value()
pylab.figure()
pylab.gray()
for i in range(num_filters_h2):
    pylab.subplot(5, 4, i+1); pylab.axis('off'); pylab.imshow(w[i,1,:,:].reshape(5,5))
#pylab.title('filters h1 learned')
pylab.savefig('project2a_2_filters_h1.png')

ind=[]
ind.append(np.random.randint(low=0, high=2000))
ind.append(np.random.randint(low=0, high=2000))
for j in ind:
    convolved0, pooled0, convolved1, pooled1 = test(teX[j:j+1,:])

    pylab.figure()
    pylab.gray()
    pylab.axis('off'); pylab.imshow(teX[j,:].reshape(28,28))
    #pylab.title('input image')
    pylab.savefig('project2a_2_input_'+str(j)+'.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_h1):
        pylab.subplot(5, 3, i+1); pylab.axis('off'); pylab.imshow(convolved0[0,i,:].reshape(20,20))
    #pylab.title('convolved feature maps h0')
    pylab.savefig('project2a_2_input_'+str(j)+'_features_h0.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_h1):
        pylab.subplot(5, 3, i+1); pylab.axis('off'); pylab.imshow(pooled0[0,i,:].reshape(10,10))
    #pylab.title('pooled feature maps p0')
    pylab.savefig('project2a_2_input_'+str(j)+'_pooled_features_h0.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_h2):
        pylab.subplot(5, 4, i+1); pylab.axis('off'); pylab.imshow(convolved1[0,i,:].reshape(6,6))
    #pylab.title('convolved feature maps h1')
    pylab.savefig('project2a_2_input_'+str(j)+'_features_h1.png')

    pylab.figure()
    pylab.gray()
    for i in range(num_filters_h2):
        pylab.subplot(5, 4, i+1); pylab.axis('off'); pylab.imshow(pooled1[0,i,:].reshape(3,3))
    #pylab.title('pooled feature maps p1')
    pylab.savefig('project2a_2_input_'+str(j)+'_pooled_features_h1.png')

pylab.show()
