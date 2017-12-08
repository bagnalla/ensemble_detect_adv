## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Alexander Bagnall

import tensorflow as tf
import numpy as np
import os
import pickle
import urllib.request
import cifar10_model as model


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3), dtype=np.float32)
    final[:, :, :, 0] = data[:, 0, :, :]
    final[:, :, :, 1] = data[:, 1, :, :]
    final[:, :, :, 2] = data[:, 2, :, :]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels


def load_batch(fpath):
    f = open(fpath, "rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 32, 32)).transpose((1, 2, 0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images), np.array(labels)


class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []

        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()

        for i in range(5):
            r, s = load_batch("cifar-10-batches-bin/data_batch_" + str(i+1) +
                              ".bin")
            train_data.extend(r)
            train_labels.extend(s)

        train_data = np.array(train_data, dtype=np.float32)
        train_labels = np.array(train_labels)

        self.test_data, self.test_labels = load_batch(
            "cifar-10-batches-bin/test_batch.bin")

        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class CIFARModel:
    def __init__(self, restore, session=None, ensemble_size=1,
                 model_dir='models/default'):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        self.ensemble_size = ensemble_size
        self.model_dir = model_dir
        self.model_names = ['m' + str(i) for i in range(ensemble_size)]
        self.session = session

    def predict(self, data):
        self.logits = [model.inference(data, name=self.model_names[i])
                       for i in range(self.ensemble_size)]
        model.load_weights(self.session, self.model_names, self.model_dir)
        self.sum_logits = tf.reduce_sum(self.logits, axis=0)
        return self.sum_logits
