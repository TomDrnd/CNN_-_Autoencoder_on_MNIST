from load import mnist
import numpy as np
import pickle
import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# 5-layer feedforward network for classification with DAE initialization for hidden layers

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

def shuffle_data (samples, target):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, target = samples[idx], target[idx]
    return samples, target

trX, teX, trY, teY = mnist()


x = T.fmatrix('x')
d = T.fmatrix('d')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

corruption_level=0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128


W1 = init_weights(28*28, 900)
b1 = init_bias(900)
b1_prime = init_bias(28*28)
W1_prime = W1.transpose()

W2 = init_weights(900, 625)
b2 = init_bias(625)
b2_prime = init_bias(900)
W2_prime = W2.transpose()

W3 = init_weights(625, 400)
b3 = init_bias(400)
b3_prime = init_bias(625)
W3_prime = W3.transpose()


#First layer
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x

y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))

params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
updates1 = [(param1, param1 - learning_rate * grad1)
           for param1, grad1 in zip(params1, grads1)]

train_da1 = theano.function(inputs=[x], outputs = [cost1], updates = updates1, allow_input_downcast = True)


#Second layer
y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
h1_to_im1 = T.nnet.sigmoid(T.dot(z2, W1_prime) + b1_prime)
cost2 = - T.mean(T.sum(x * T.log(h1_to_im1) + (1 - x) * T.log(1 - h1_to_im1), axis=1))

params2 = [W2, b2, b2_prime]
grads2 = T.grad(cost2, params2)
updates2 = [(param2, param2 - learning_rate * grad2)
           for param2, grad2 in zip(params2, grads2)]

train_da2 = theano.function(inputs=[x], outputs = [cost2], updates = updates2, allow_input_downcast = True)


#Third layer
y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
h2_to_h1 = T.nnet.sigmoid(T.dot(z3, W2_prime) + b2_prime)
h1_to_im2 = T.nnet.sigmoid(T.dot(h2_to_h1, W1_prime) + b1_prime)
cost3 = - T.mean(T.sum(x * T.log(h1_to_im2) + (1 - x) * T.log(1 - h1_to_im2), axis=1))

params3 = [W3, b3, b3_prime]
grads3 = T.grad(cost3, params3)
updates3 = [(param3, param3 - learning_rate * grad3)
           for param3, grad3 in zip(params3, grads3)]

train_da3 = theano.function(inputs=[x], outputs = [cost3], updates = updates3, allow_input_downcast = True)


#Classification layer
W4 = init_weights(400, 10)
b4 = init_bias(10)
y1_class = T.nnet.sigmoid(T.dot(x, W1) + b1)
y2_class = T.nnet.sigmoid(T.dot(y1_class, W2) + b2)
y3_class = T.nnet.sigmoid(T.dot(y2_class, W3) + b3)
p_y2 = T.nnet.softmax(T.dot(y3_class, W4)+b4)
out = T.argmax(p_y2, axis=1)
cost4 = T.mean(T.nnet.categorical_crossentropy(p_y2, d))

params4 = [W1, b1, W2, b2, W3, b3, W4, b4]
grads4 = T.grad(cost4, params4)
updates4 = [(param4, param4 - learning_rate * grad4)
           for param4, grad4 in zip(params4, grads4)]
train_ffn = theano.function(inputs=[x, d], outputs = cost4, updates = updates4, allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = out, allow_input_downcast=True)



#Training autoencoder

print('training dae1 ...')
d = []
for epoch in range(training_epochs):
    #go through training set
    c = []
    trX, trY =shuffle_data(trX, trY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train_da1(trX[start:end])
        c.append(cost)
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_2_h1.png')

print('training dae2 ...')
d = []
for epoch in range(training_epochs):
    #go through training set
    c = []
    trX, trY =shuffle_data(trX, trY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train_da2(trX[start:end])
        c.append(cost)
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_2_h2.png')

print('training dae3 ...')
d = []
for epoch in range(training_epochs):
    #go through training set
    c = []
    trX, trY =shuffle_data(trX, trY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train_da3(trX[start:end])
        c.append(cost)
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])

pylab.figure()
pylab.plot(range(training_epochs), d)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_2_h3.png')


#Classification

#just use a subsample of the dataset to compare equally with the CNN for 100 iterations
#trX, trY = trX[:12000], trY[:12000]
#teX, teY = teX[:2000], teY[:2000]
training_epochs=100

print('\ntraining ffn ...')
d, a = [], []
for epoch in range(training_epochs):
    if epoch%10==0:
        print(epoch)
    c = []
    trX, trY =shuffle_data(trX, trY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_ffn(trX[start:end], trY[start:end]))
    d.append(np.mean(c, dtype='float64'))
    a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
    print(a[epoch])

#This part will only work if at least the project2a_1.py has been executed

#acc = pickle.load(file('accuracy_mnist.pickle'))
#pylab.plot(range(training_epochs), a, '-y', label='DAE')
#pylab.savefig('project_2b_2_accuracy.png')
#pickle.dump(acc, file('accuracy_mnist.pickle', 'w'))

#acc = pickle.load(file('cost_mnist.pickle'))
#pylab.plot(range(training_epochs), d, '-y', label='DAE')
#pylab.savefig('project_2b_2_error.png')
#pickle.dump(acc, file('cost_mnist.pickle', 'w'))

pylab.figure()
pylab.plot(range(training_epochs), a, '-y', label='DAE')
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.savefig('project_2b_2_accuracy.png')

pylab.figure()
pylab.plot(range(training_epochs), d, '-y', label='DAE')
pylab.xlabel('epochs')
pylab.ylabel('cross-entropy')
pylab.savefig('project_2b_2_error.png')


#Just to see if many evolutions occur but it's not really the case (and that's good)
w3 = W3.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(25,25))
pylab.savefig('figure_2b_2_weights_h3.png')

w2 = W2.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(30,30))
pylab.savefig('figure_2b_2_weights_h2.png')

w1 = W1.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig('figure_2b_2_weights_h1.png')

pylab.show()
