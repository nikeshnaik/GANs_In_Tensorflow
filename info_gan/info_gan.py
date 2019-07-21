
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def sample_c(m):
    return np.random.multinomial(1, 10*[0.1], size=m)


def generator(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    with tf.variable_scope('generator'):

        init = tf.keras.initializers.glorot_normal()
        h1 = tf.layers.dense(inputs=inputs,units=256,activation=tf.nn.relu,kernel_initializer=init,name='1st_Layer',use_bias=True)
        h2 = tf.layers.dense(inputs=h1,units=256,activation=tf.nn.relu,kernel_initializer=init,name='2nd_Layer',use_bias=True)
        # h2 = tf.layers.dense(inputs=h2,units=512,activation=tf.nn.relu,kernel_initializer=init,name='22nd_Layer',use_bias=True)
        img = tf.layers.dense(inputs=h2,units=784,activation=tf.nn.tanh,kernel_initializer=init,name='3rd_Layer',use_bias=True)

    return img


def discriminator(x):
    with tf.variable_scope('discriminator'):

        init = tf.keras.initializers.glorot_normal()
        h1 = tf.layers.dense(inputs=x,units=512,activation=tf.nn.leaky_relu,kernel_initializer=init,use_bias=True,name='1st_dense')
        h2 = tf.layers.dense(inputs=h1,units=256,activation=tf.nn.leaky_relu,kernel_initializer=init,use_bias=True,name='2nd_dense')
        logits = tf.layers.dense(inputs=h2,units=1,activation=tf.nn.sigmoid,kernel_initializer=init,name='3rd_dense',use_bias=True)


    return logits

def Q(x):

    with tf.variable_scope('latent_code'):
        init = tf.keras.initializers.glorot_normal()

        h1 = tf.layers.dense(inputs=x,units=128,activation=tf.nn.relu,kernel_initializer=init,use_bias=True,name='1st_dense')
        h1 = tf.layers.dense(inputs=h1,units=64,activation=tf.nn.relu,kernel_initializer=init,use_bias=True,name='11st_dense')
        h2 = tf.layers.dense(inputs=h1,units=10,activation=tf.nn.softmax,kernel_initializer=init,use_bias=True,name='2nd_dense')

    return h2

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

X = tf.placeholder(tf.float32, shape=[None, 784])

Z = tf.placeholder(tf.float32, shape=[None, 16])
c = tf.placeholder(tf.float32, shape=[None, 10])

G_sample = generator(Z, c)

with tf.variable_scope("") as scope:
    D_real = discriminator(X)
    scope.reuse_variables()
    D_fake = discriminator(G_sample)

Q_c_given_x = Q(G_sample)

D_loss = - tf.reduce_mean(tf.log(D_real + 1e-8) + tf.log(1 - D_fake + 1e-8))
G_loss = - tf.reduce_mean(tf.log(D_fake + 1e-8))


Q_loss  = tf.reduce_mean(- tf.reduce_sum(tf.log(Q_c_given_x + 1e-8) * c, 1))


D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
Q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'latent_code')


D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(D_loss,var_list=D_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(G_loss,var_list=G_vars)
Q_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(Q_loss,var_list=G_vars+Q_vars)

mb_size = 32
Z_dim = 16


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for it in range(1000000):
        if it % 1000 == 0:
            Z_noise = sample_Z(16, Z_dim)

            idx = np.random.randint(0, 10)
            c_noise = np.zeros([16, 10])
            c_noise[range(16), idx] = 1

            samples = sess.run(G_sample,
                               feed_dict={Z: Z_noise, c: c_noise})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, _ = mnist.train.next_batch(mb_size)
        Z_noise = sample_Z(mb_size, Z_dim)
        c_noise = sample_c(mb_size)

        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={X: X_mb, Z: Z_noise, c: c_noise})

        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={Z: Z_noise, c: c_noise})

        _,Q_loss_curr = sess.run([Q_solver, Q_loss], feed_dict={Z: Z_noise, c: c_noise})

        if it % 10 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('Q_loss: {:.4}'.format(Q_loss_curr))
            print()
