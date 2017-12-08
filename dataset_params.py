# This file provides dataset-specific parameters and functions for MNIST
# and CIFAR10.

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

def make_dataset(images, labels):
    return DataSet(images, labels, reshape=False, dtype=tf.uint8)


# MNIST

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import mnist_model as mnist_model
from util import save_mnist_images as mnist_save_images

MNIST_NUM_CLASSES = 10
MNIST_IMAGE_SIZE = 28

def mnist_example_shape(batch_size):
    return (batch_size, MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE)

def mnist_load_data():
    data_sets = input_data.read_data_sets('data')
    return data_sets.train, data_sets.validation, data_sets.test

########################################################################


## CIFAR10

import cifar10_model as cifar10_model
import download_cifar10 as dl
from util import save_cifar10_images as cifar10_save_images

CIFAR10_NUM_CLASSES = 10
CIFAR10_IMAGE_SIZE = 32

def cifar10_example_shape(batch_size):
    return (batch_size, CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE, 3)

def cifar10_load_data():
    dl.maybe_download_and_extract()
    train_data = dl.load_training_data()
    train_images, train_labels, train_one_hot_labels = train_data
    train_data_set = make_dataset(train_images, train_labels)
    validation_data_set = make_dataset(np.array([]), np.array([]))
    test_data = dl.load_test_data()
    test_images, test_labels, test_one_hot_labels = test_data
    test_data_set = make_dataset(test_images, test_labels)
    return train_data_set, validation_data_set, test_data_set

########################################################################


# Choose either MNIST or CIFAR10
def choose_dataset(set_name):
    if set_name == 'MNIST':
        return mnist_model, mnist_save_images, MNIST_NUM_CLASSES, \
            MNIST_IMAGE_SIZE, mnist_example_shape, mnist_load_data
    else:
        return cifar10_model, cifar10_save_images, CIFAR10_NUM_CLASSES, \
            CIFAR10_IMAGE_SIZE, cifar10_example_shape, cifar10_load_data
