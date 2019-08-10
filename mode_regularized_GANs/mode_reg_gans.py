import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os



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

def log(x):
    return tf.log(x + 1e-8)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):

    with tf.variable_scope("generator"):

        init = tf.keras.initializers.glorot_normal()
        h1 = tf.layers.dense(inputs=z,units=128,activation=tf.nn.relu,kernel_initializer=init,name='1st_Layer',use_bias=True)
        img = tf.layers.dense(inputs=h1,units=784,activation=tf.nn.sigmoid,kernel_initializer=init,name='2nd_Layer',use_bias=True)

    return img

def discriminator(x):

    with tf.variable_scope("discriminator"):

        init = tf.keras.initializers.glorot_normal()
        h1 = tf.layers.dense(inputs=x,units=128,activation=tf.nn.leaky_relu,kernel_initializer=init,use_bias=True,name='1st_dense')
        out = tf.layers.dense(inputs=h1,units=1,kernel_initializer=init,activation=tf.nn.sigmoid,name='3rd_dense',use_bias=True)

    return out

def encoder(x):

    with tf.variable_scope("encoder"):

        init = tf.keras.initializers.glorot_normal()
        h1 = tf.layers.dense(inputs=x,units=128,activation=tf.nn.leaky_relu,kernel_initializer=init,use_bias=True,name='1st_dense')
        out = tf.layers.dense(inputs=h1,units=10,kernel_initializer=init,name='3rd_dense',use_bias=True)

    return out

batch_size =32
X_dim = 784
z_dim = 10
h_dim = 128
lam1 = 1e-2
lam2 = 1e-2

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

encoder_output = encoder(X)

with tf.variable_scope('') as scope:
    g_sample = generator(z)
    scope.reuse_variables()
    g_sample_reg = generator(encoder_output)

with tf.variable_scope("") as scope:

    D_real = discriminator(X)
    scope.reuse_variables()
    D_fake = discriminator(g_sample)
    scope.reuse_variables()
    D_reg = discriminator(g_sample_reg)

mse = tf.reduce_sum((X-g_sample_reg)**2,1)

D_loss = -tf.reduce_mean(log(D_real)+log(1-D_fake))
E_loss = -tf.reduce_mean(lam1*mse+lam2*log(D_reg))
G_loss = -tf.reduce_mean(log(D_fake)) + E_loss

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')


D_solver = (tf.train.AdamOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=D_vars))
G_solver = (tf.train.AdamOptimizer(learning_rate=1e-3).minimize(G_loss, var_list=G_vars))
E_solver = (tf.train.AdamOptimizer(learning_rate=1e-3).minimize(E_loss, var_list=E_vars))


if not os.path.exists('out/'):
    os.makedirs('out/')


i = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for it in range(1000000):
        X_batch,_ = mnist.train.next_batch(batch_size)

        _,D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X:X_batch,z:sample_z(batch_size,z_dim)})
        _,G_loss_curr = sess.run([G_solver, G_loss],feed_dict={X:X_batch,z:sample_z(batch_size,z_dim)})
        _,E_loss_curr = sess.run([E_solver, E_loss],feed_dict={X:X_batch,z:sample_z(batch_size,z_dim)})

        if it % 1000 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}'
                  .format(it, D_loss_curr, G_loss_curr, E_loss_curr))

            samples = sess.run(g_sample, feed_dict={z: sample_z(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
