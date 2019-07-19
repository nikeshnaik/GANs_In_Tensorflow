import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../MNIST_data',one_hot=True)


def show_images(images,it):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    plt.savefig('out/{}.png'.format(it),bbox_inches='tight')
    plt.close(fig)


def sample_noise(batch_size, dim):
    return tf.random_uniform(shape=(batch_size,dim),minval=-1,maxval=1)

def discriminator(x):

    with tf.variable_scope('discriminator'):
        init = tf.keras.initializers.glorot_normal()
        h1 = tf.layers.dense(inputs=x,units=256,activation=tf.nn.leaky_relu,kernel_initializer=init,use_bias=True,name='1st_dense')
        h2 = tf.layers.dense(inputs=h1,units=256,activation=tf.nn.leaky_relu,kernel_initializer=init,use_bias=True,name='2nd_dense')
        logits = tf.layers.dense(inputs=h2,units=1,kernel_initializer=init,name='3rd_dense',use_bias=True)

    return logits

def generator(z):

    with tf.variable_scope("generator"):

        init = tf.keras.initializers.glorot_normal()
        h1 = tf.layers.dense(inputs=z,units=1024,activation=tf.nn.relu,kernel_initializer=init,name='1st_Layer',use_bias=True)
        h2 = tf.layers.dense(inputs=h1,units=1024,activation=tf.nn.relu,kernel_initializer=init,name='2nd_Layer',use_bias=True)
        img = tf.layers.dense(inputs=h2,units=784,activation=tf.nn.tanh,kernel_initializer=init,name='3rd_Layer',use_bias=True)

    return img

def gan_loss(logits_real, logits_fake):
    D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
    D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),logits=logits_fake))
    D_loss = D_fake + D_real
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),logits=logits_fake))

    return D_loss, G_loss

def get_solvers(learning_rate=0.0005):

    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return D_solver, G_solver


tf.reset_default_graph()

# number of images for each batch
batch_size = 128
# our noise dimension
noise_dim = 96
# placeholder for images from the training dataset
x = tf.placeholder(tf.float32, [None, 784])
# random noise fed into our generator
z = sample_noise(batch_size, noise_dim)
# generated images
G_sample = generator(z)

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real = discriminator(x)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

# get our solver
D_solver, G_solver = get_solvers()

# get our loss
D_loss, G_loss = gan_loss(logits_real, logits_fake)

# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

# D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
# G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')


if not os.path.exists('out/'):
    os.makedirs('out/')

num_epoch = 50

max_iter = int(mnist.train.num_examples*num_epoch/batch_size)

# compute the number of iterations we need
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for it in range(max_iter):
        # every show often, show a sample result
        if it % 500== 0:
            samples = sess.run(G_sample)

            show_images(samples[:16],it)
            # plt.show()
        # run a batch of data through the network
        minibatch,minbatch_y = mnist.train.next_batch(batch_size)
        _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: minibatch})
        _, G_loss_curr = sess.run([G_train_step, G_loss])

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if it % 100 == 0:
            print('Iteration: {}, D Loss: {:.4}, G Loss:{:.4}'.format(it,D_loss_curr,G_loss_curr))

    samples = sess.run(G_sample)
    fig = show_images(samples[:16],'final')
    plt.show()
