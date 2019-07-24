import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


batch_size = 32
X_dim = 784
z_dim = 10
h_dim = 128

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

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
        out = tf.layers.dense(inputs=h1,units=1,kernel_initializer=init,name='3rd_dense',use_bias=True)

    return out


X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_sample = generator(z)

with tf.variable_scope("") as scope:
    D_real = discriminator(X)
    scope.reuse_variables()
    D_fake = discriminator(G_sample)


D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')


D_solver = (tf.train.AdamOptimizer().minimize(-D_loss, var_list=D_vars))
G_solver = (tf.train.AdamOptimizer().minimize(G_loss, var_list=G_vars))


clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D_vars]

if not os.path.exists('out/'):
    os.makedirs('out/')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    for it in range(1000000):
        for _ in range(5):
            x_batch, _ = mnist.train.next_batch(batch_size)
            _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D],feed_dict={X: x_batch, z: sample_z(batch_size, z_dim)})

        _, G_loss_curr = sess.run([G_solver, G_loss],feed_dict={z: sample_z(batch_size, z_dim)})

        if it % 100 == 0:
            print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(it, D_loss_curr, G_loss_curr))

            if it % 1000 == 0:
                samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)
