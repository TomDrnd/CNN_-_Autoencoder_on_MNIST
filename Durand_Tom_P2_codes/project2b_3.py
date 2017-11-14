from load import mnist
import numpy as np
import pickle
import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# 5-layer feedforward network for classification with DAE initialization for hidden layers
#with sparsity constraints and momentum

corruption_level=0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128
momentum = 0.1
penalty = 0.5
sparcity = 0.05


def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_velocity(n_visible, n_hidden):
    initial_V = np.zeros((n_visible, n_hidden), dtype=theano.config.floatX)
    return theano.shared(value=initial_V, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

def sgd(cost, params, velocity):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    #updates for velocity and weights
    for v, g, p in zip(velocity, grads[0::2], params[0::2]):
        updates.append([v, momentum*v - g*learning_rate])
        updates.append([p, p+momentum*v-g*learning_rate])
    #updates for biases
    for g, p in zip(grads[1::2], params[1::2]):
        updates.append([p, p - g*learning_rate])
    return updates

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


W1 = init_weights(28*28, 900)
V1 = init_velocity(28*28, 900)
b1 = init_bias(900)
b1_prime = init_bias(28*28)
W1_prime = W1.transpose()

W2 = init_weights(900, 625)
V2 = init_velocity(900, 625)
b2 = init_bias(625)
b2_prime = init_bias(900)
W2_prime = W2.transpose()

W3 = init_weights(625, 400)
V3 = init_velocity(625, 400)
b3 = init_bias(400)
b3_prime = init_bias(625)
W3_prime = W3.transpose()


#First layer
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x

y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1)) \
			+ penalty*T.shape(y1)[1]*(sparcity*T.log(sparcity) + (1-sparcity)*T.log(1-sparcity)) \
			- penalty*sparcity*T.sum(T.log(T.mean(y1, axis=0))) \
			- penalty*(1-sparcity)*T.sum(T.log(1-T.mean(y1, axis=0)))

params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
updates1 = [(V1, momentum*V1 - grads1[0]*learning_rate),
            (W1, W1+momentum*V1 - grads1[0]*learning_rate),
            (b1, b1 - grads1[1]*learning_rate),
            (b1_prime, b1_prime - grads1[2]*learning_rate)]

train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)


#Second layer
y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
h1_to_im1 = T.nnet.sigmoid(T.dot(z2, W1_prime) + b1_prime)
cost2 = - T.mean(T.sum(x * T.log(h1_to_im1) + (1 - x) * T.log(1 - h1_to_im1), axis=1))\
			+ penalty*T.shape(y2)[1]*(sparcity*T.log(sparcity) + (1-sparcity)*T.log(1-sparcity)) \
			- penalty*sparcity*T.sum(T.log(T.mean(y2, axis=0))) \
			- penalty*(1-sparcity)*T.sum(T.log(1-T.mean(y2, axis=0)))

params2 = [W2, b2, b2_prime]
grads2 = T.grad(cost2, params2)
updates2 = [(V2, momentum*V2 - grads2[0]*learning_rate),
            (W2, W2+momentum*V2 - grads2[0]*learning_rate),
            (b2, b2 - grads2[1]*learning_rate),
            (b2_prime, b2_prime - grads2[2]*learning_rate)]

train_da2 = theano.function(inputs=[x], outputs = cost2, updates = updates2, allow_input_downcast = True)


#Third layer
y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
h2_to_h1 = T.nnet.sigmoid(T.dot(z3, W2_prime) + b2_prime)
h1_to_im2 = T.nnet.sigmoid(T.dot(h2_to_h1, W1_prime) + b1_prime)
cost3 = - T.mean(T.sum(x * T.log(h1_to_im2) + (1 - x) * T.log(1 - h1_to_im2), axis=1))\
			+ penalty*T.shape(y3)[1]*(sparcity*T.log(sparcity) + (1-sparcity)*T.log(1-sparcity)) \
			- penalty*sparcity*T.sum(T.log(T.mean(y3, axis=0))) \
			- penalty*(1-sparcity)*T.sum(T.log(1-T.mean(y3, axis=0)))

params3 = [W3, b3, b3_prime]
grads3 = T.grad(cost3, params3)
updates3 = [(V3, momentum*V3 - grads3[0]*learning_rate),
            (W3, W3+momentum*V3 - grads3[0]*learning_rate),
            (b3, b3 - grads3[1]*learning_rate),
            (b3_prime, b3_prime - grads3[2]*learning_rate)]

train_da3 = theano.function(inputs=[x], outputs = cost3, updates = updates3, allow_input_downcast = True)

test = theano.function(inputs=[x], outputs = [y1, y2, y3, h1_to_im2], allow_input_downcast = True)


#Classification
W4 = init_weights(400, 10)
V4 = init_velocity(400, 10)
b4 = init_bias(10)
y1_class = T.nnet.sigmoid(T.dot(x, W1) + b1)
y2_class = T.nnet.sigmoid(T.dot(y1_class, W2) + b2)
y3_class = T.nnet.sigmoid(T.dot(y2_class, W3) + b3)
p_y2 = T.nnet.softmax(T.dot(y3_class, W4)+b4)
out = T.argmax(p_y2, axis=1)
cost4 = T.mean(T.nnet.categorical_crossentropy(p_y2, d))

params4 = [W1, b1, W2, b2, W3, b3, W4, b4]
grads4 = T.grad(cost4, params4)
updates4 = sgd(cost4, params4, [V1, V2, V3, V4])

train_ffn = theano.function(inputs=[x, d], outputs = cost4, updates = updates4, allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = out, allow_input_downcast=True)


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
pylab.savefig('figure_2b_3_h1.png')

w1 = W1.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig('figure_2b_3_weights_h1.png')


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
pylab.savefig('figure_2b_3_h2.png')

w2 = W2.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(30,30))
pylab.savefig('figure_2b_3_weights_h2.png')


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
pylab.savefig('figure_2b_3_h3.png')

w3 = W3.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(25,25))
pylab.savefig('figure_2b_3_weights_h3.png')


act1, act2, act3, reconstructed_im = test(teX[:100])


pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(reconstructed_im[i,:].reshape(28,28))
pylab.savefig('reconstructed_images_spars.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(act1[i,:].reshape(30,30))
pylab.savefig('activation_spars1.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(act2[i,:].reshape(25,25))
pylab.savefig('activation_spars2.png')

pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(act3[i,:].reshape(20,20))
pylab.savefig('activation_spars3.png')


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

#use the previous figure to compare the methods
#It will only work if at least the project2a_1.py has been executed

#acc = pickle.load(file('accuracy_mnist.pickle'))
#pylab.plot(range(training_epochs), a, '-k', label='DAE_sparcity')
#pylab.savefig('project_2b_3_accuracy.png')
#pickle.dump(acc, file('accuracy_mnist.pickle', 'w'))

#acc = pickle.load(file('cost_mnist.pickle'))
#pylab.plot(range(training_epochs), d, '-k', label='DAE_sparcity')
#pylab.savefig('project_2b_3_cost.png')
#pickle.dump(acc, file('cost_mnist.pickle', 'w'))

pylab.figure()
pylab.plot(range(training_epochs), d, '-k', label='DAE_sparcity')
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_3_cost.png')

pylab.figure()
pylab.plot(range(training_epochs), a, '-k', label='DAE_sparcity')
pylab.xlabel('iterations')
pylab.ylabel('test accuracy')
pylab.savefig('figure_2b_3_accuracy.png')


pylab.show()
