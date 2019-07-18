import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mnsit = input_data('../../MNIST_data',one_hot=True)
batch_size=64
Z_dim =100
X_dim = mnist.train.images.shape[1]
Y_dim = mnist.train.labels.shape[1]
h_dim = 128
