import sys

print(sys.argv[2])

import pandas as pd
from sklearn.model_selection import train_test_split
import os

import numpy as np
np.random.seed(123)
print("NumPy:{}".format(np.__version__))

import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,10
print("Matplotlib:{}".format(mpl.__version__))

import tensorflow as tf
tf.set_random_seed(123)
print("TensorFlow:{}".format(tf.__version__))

import keras
print("Keras:{}".format(keras.__version__))


DATA = pd.read_table(sys.argv[1], header=0, sep="\s+").astype(np.float32)

Class = pd.read_table("~/Class190314.txt", header=0, sep="\s+")

trX, teX, trY, teY = train_test_split(DATA, Class, test_size=0.25)


vars = trX.var(axis = 0)
VFiltered = vars.nlargest(1024)
trX = trX.loc[:, VFiltered.index]

teX = teX.loc[:, VFiltered.index]


n_visible = 1024
n_nodes = [n_visible, 900, 800, 750]
corruption_level = 0.3
hidden_size = len(n_nodes)-1
Z = [None]*hidden_size #Estimated output
cost = [None]*hidden_size
train_op = [None]*hidden_size #trainning operation
n_classes = 2

# create node for input data
X = tf.placeholder("float", name='X')

#saver = tf.train.Saver()


weights_encoder=dict()
weights_decoder=dict()
biases_encoder=dict()
biases_decoder=dict()
for i in range(hidden_size): #initialize variables for each hidden layer
    W_init_max = 4 * np.sqrt(6. / (n_nodes[i] + n_nodes[i+1])) #initialize variables with random values
    W_init = tf.random_uniform(shape=[n_nodes[i], n_nodes[i+1]],
                                minval=-W_init_max,
                                maxval=W_init_max)
    weights_encoder[i] = tf.Variable(W_init, name='weight_encoder')
    weights_decoder[i] = tf.transpose(weights_encoder[i], name='weight_decoder') #decoder weights are tied with encoder size
    biases_encoder[i] = tf.Variable(tf.random_normal([n_nodes[i+1]]), name='bias_encoder')
    biases_decoder[i] = tf.Variable(tf.random_normal([n_nodes[i]]), name='bias_decoder')


def corruption(input): #corruption of the input
    mask = np.random.binomial(1, 1 - corruption_level,input.shape ) #mask with several zeros at certain position
    corrupted_input = input*mask
    return corrupted_input


def model(input, W, b, W_prime, b_prime):
    Y = tf.nn.leaky_relu(tf.matmul(input, W) + b)  # hidden state
    Z = tf.nn.leaky_relu(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input
    return Z

def encode(input, W, b, n):
    if n == 0:
        Y = input #input layer no encode needed
    else:
        for i in range(n): #encode the input layer by layer
            Y = tf.nn.leaky_relu(tf.add(tf.matmul(input, W[i]), b[i]))
            input = Y #output become input for next layer encode
        Y = Y.eval() #convert tensor.object to ndarray
    return Y



def decode(input, W_prime, b_prime, n):
    if n == 0:
        Y = input
    else:
        for i in range(n):
            Y = tf.nn.leaky_relu(tf.add(tf.matmul(input, W_prime[n-i-1]), b_prime[n-i-1]))
            input = Y
            Y = Y.eval()
    return Y


for i in range(hidden_size): #how many layers need to be trained
    Z[i] = model(X, weights_encoder[i],  biases_encoder[i], weights_decoder[i], biases_decoder[i])
    cost[i] = tf.reduce_sum(tf.pow(X - Z[i], 2))
    train_op[i] = tf.train.RMSPropOptimizer(learning_rate=0.00001).minimize(cost[i])

##Classifier
input_X = tf.placeholder(dtype='float32', shape=(trX.shape[0], trX.shape[1]),name="input_X")
input_y = tf.placeholder(dtype='int32', name="input_y")


W_init_max = 4 * np.sqrt(6. / (n_nodes[-1] + n_classes)) #initialize variables with random values
W_init = tf.random_uniform(shape=[n_nodes[-1], n_classes],
                                minval=-W_init_max,
                                maxval=W_init_max)
W = tf.Variable(W_init, name = 'W')
b = tf.Variable(tf.zeros([n_classes]), name='b')


#one_hot_trY = np.eye(2)[input_y.iloc[:,:0]]
one_hot_labels = input_y #tf.one_hot(input_y, depth=n_classes, axis = -1)
Cl_input = tf.placeholder(dtype='float32', name='Cl_input') #, shape=(trX.shape[0], n_nodes[-1])
output_logits = tf.matmul(Cl_input, W) + b       #####CONNECT hidden_layer_output ..... how?
Cost = tf.losses.softmax_cross_entropy(one_hot_labels, output_logits)
Train_Op = tf.train.RMSPropOptimizer(learning_rate=0.00001).minimize(Cost, var_list=[W, b])
predictedClass = tf.nn.softmax(output_logits)

#classifier = tf.nn.softmaxZ[4]

TRY = trY
TRY[TRY == "Perturbed"] = 1
TRY[TRY == "Control"] = 0
TRY = tf.one_hot(TRY.astype('uint8'), depth = 2)
TRY = tf.cast(TRY, tf.int32)


saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # initialise all variables
    tf.global_variables_initializer().run()
    for j in range(hidden_size):
        encoded_trX = encode(trX, weights_encoder, biases_encoder, j)
        encoded_teX = encode(teX, weights_encoder, biases_encoder, j)
        for i in range(2500):
            for start, end in zip(range(0, len(trX), 50), range(50, len(trX), 50)):
                input_ = encoded_trX[start:end]
                sess.run(train_op[j], feed_dict={X: corruption(input_)})
                print("Layer:", j, i, sess.run(cost[j], feed_dict={X: encoded_teX}))
    ForC = encode(trX, weights_encoder, biases_encoder, hidden_size)
    Class = sess.run(TRY)
    for i in range(2500):
        for start, end in zip(range(0, len(trX), 50), range(50, len(trX), 50)):
            input_ = ForC[start:end]
            input_Y = Class[start:end]
            sess.run(Train_Op, feed_dict={Cl_input: input_, input_y: input_Y})
            print("Classifying:", i, sess.run(Cost, feed_dict={Cl_input: ForC, input_y: Class}))
    saver.save(sess, sys.argv[2])
    test = encode(teX, weights_encoder, biases_encoder, hidden_size)
    teX_pred_class = sess.run(tf.nn.softmax(tf.matmul(test, W) + b))


pd.DataFrame(teX).to_csv(sys.argv[3])
pd.DataFrame(teY).to_csv(sys.argv[4])

pd.DataFrame(trX).to_csv(sys.argv[5])
pd.DataFrame(trY).to_csv(sys.argv[6])


