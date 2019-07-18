import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

modules = ['tensorflow','numpy','matplotlib','os']
if not all([True if each in sys.modules else False for each in modules]):
    raise ModuleNotFoundError


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1./tf.sqrt(in_dim/2.)
    return tf.random_normal(shape=size,stddev=xavier_stddev)


X = tf.placeholder(tf.float32,shape=[None,784])

Discr_W1 = tf.Variable(xavier_init([784,128]))
Discr_b1 = tf.Variable(tf.zeros(shape=[128]))

Discr_W2 = tf.Variable(xavier_init([128,1]))
Discr_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_Discr = [Discr_W1, Discr_W2, Discr_b1, Discr_b2]

Z = tf.placeholder(tf.float32,shape=[None,100])

Gene_W1 = tf.Variable(xavier_init([100,128]))
Gene_b1 = tf.Variable(tf.zeros(shape=[128]))

Gene_W2 = tf.Variable(xavier_init([128,784]))
Gene_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_Gene = [Gene_W1, Gene_b1, Gene_W2, Gene_b2]


def sample_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])

def generator(z):
    Gene_h1 = tf.nn.relu(tf.matmul(z,Gene_W1)+ Gene_b1)
    Gene_log_prob = tf.matmul(Gene_h1,Gene_W2) + Gene_b2
    Gene_prob = tf.nn.sigmoid(Gene_log_prob)

    return Gene_prob

def discriminator(x):
    Discr_h1 = tf.nn.relu(tf.matmul(x, Discr_W1)+Discr_b1)
    Discr_logit = tf.matmul(Discr_h1, Discr_W2) + Discr_b2
    Discr_prob = tf.nn.sigmoid(Discr_logit)

    return Discr_prob, Discr_logit

def plot(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)

    for i,sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28),cmap='Greys_r')

    return fig


Gene_Sample = generator(Z)
Discr_real, Discr_logit_real = discriminator(X)
Discr_fake, Discr_logit_fake = discriminator(Gene_Sample)


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Discr_logit_real,labels=tf.ones_like(Discr_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Discr_logit_fake,labels=tf.zeros_like(Discr_logit_fake)))

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Discr_logit_fake,labels=tf.ones_like(Discr_logit_fake)))

Dis_optimizer = tf.train.AdamOptimizer().minimize(D_loss,var_list=theta_Discr)
Gen_optimizer = tf.train.AdamOptimizer().minimize(G_loss,var_list=theta_Gene)

batch_size= 128
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data',one_hot=True)

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for it in range(100000):
        if it%100 ==0:
            sample = sess.run(Gene_Sample,feed_dict={Z:sample_Z(16l,Z_dim)})

            fig = plot(sample)

            plt.savefig('out/{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
            i+=1
            plt.close(fig)

        X_batch, _ = mnist.train.next_batch(batch_size)

        _,D_loss_curr = sess.run([Dis_optimizer, D_loss], feed_dict={X:X_batch, Z:sample_Z(batch_size,Z_dim)})
        _, G_loss_curr = sess.run([Gen_optimizer, G_loss], feed_dict={Z:sample_Z(batch_size,Z_dim)})

        if it%100==0:
            print("Iter: {}".format(it))
            print("D_loss: {:.4}".format(D_loss_curr))
            print("G_loss: {:.4}".format(G_loss_curr))
            print()
