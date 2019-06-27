import sys

import os
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


DATA = pd.read_csv(sys.argv[1], header=0)#.astype(np.float32)
Class = pd.read_csv(sys.argv[2], header=0)

DATA = DATA.iloc[:,1:1025]
Class = Class.iloc[:,1]

trX, teX, trY, teY = train_test_split(DATA, Class, test_size=0.5)


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(sys.argv[3])
    new_saver.restore(sess, tf.train.latest_checkpoint('/home/fwhite'))
    stuff = tf.trainable_variables()
    for i in stuff: 
      print(i, end="\n")
    print(sess.run(stuff[0]).shape)

print(sys.argv[3])


weights = dict()
weights[0] = stuff[0]
weights[1] = stuff[3]
weights[2] = stuff[6]
weights[3] = stuff[9]

biases = dict()
biases[0] = stuff[1]
biases[1] = stuff[4]
biases[2] = stuff[7]
biases[3] = stuff[10]

X = tf.placeholder("float", name='X')
input_y = tf.placeholder(dtype='int32', name="input_y")
one_hot_labels = input_y

def encoder(X, weights, biases):
    L1 = tf.nn.leaky_relu(tf.matmul(X, weights[0]) + biases[0])
    L2 = tf.nn.leaky_relu(tf.matmul(L1, weights[1]) + biases[1])
    L3 = tf.nn.leaky_relu(tf.matmul(L2, weights[2]) + biases[2])
    return L3



def finetune(X, weights, biases):
    L1 = tf.nn.leaky_relu(tf.matmul(X, weights[0]) + biases[0])
    L2 = tf.nn.leaky_relu(tf.matmul(L1, weights[1]) + biases[1])
    L3 = tf.nn.leaky_relu(tf.matmul(L2, weights[2]) + biases[2])
    output_logits = tf.matmul(L3, weights[3]) + biases[3]
    return output_logits


# build model graph
Z = finetune(X, weights, biases)

#OL = tf.nn.softmax(Z)


CostF = tf.losses.softmax_cross_entropy(one_hot_labels, Z)
Train_OpF = tf.train.RMSPropOptimizer(learning_rate=0.00001, name='FT').minimize(CostF, var_list=[weights[0], weights[1], weights[2],
                                                                                     weights[3], biases[0], biases[1],
                                                                                     biases[2], biases[3]])

TRY = trY
TRY[TRY == "Perturbed"] = 1
TRY[TRY == "Control"] = 0
TRY = tf.one_hot(TRY.astype('uint8'), depth = 2)
TRY = tf.cast(TRY, tf.int32)

saver = tf.train.import_meta_graph(sys.argv[3])


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(sys.argv[3])
    new_saver.restore(sess, tf.train.latest_checkpoint('/home/fwhite'))
    tf.initialize_all_variables().run()
    Class = sess.run(TRY)
    for i in range(1000):
        for start, end in zip(range(0, len(trX), 50), range(50, len(trX), 50)):
            input_ = trX[start:end]
            input_Y = Class[start:end]
            sess.run(Train_OpF, feed_dict={X: input_, input_y: input_Y})
        print(i, sess.run(CostF, feed_dict={X: trX, input_y: Class}))
    test = sess.run(encoder(teX.astype('float32'), weights, biases))
    teX_pred_class_NS = sess.run(finetune(teX.astype('float32'), weights, biases))
    teX_pred_class = sess.run(tf.nn.softmax(teX_pred_class_NS))

pd.DataFrame(teX).to_csv(sys.argv[4])
pd.DataFrame(teY).to_csv(sys.argv[5])
pd.DataFrame(teX_pred_class).to_csv(sys.argv[6])
pd.DataFrame(test).to_csv(sys.argv[7])


