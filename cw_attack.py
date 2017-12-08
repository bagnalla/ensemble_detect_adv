##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified by Alexander Bagnall

import tensorflow as tf
import numpy as np
import gzip, pickle
import time

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from l2_attack import CarliniL2


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


# num_samples should be a multiple of the batch size
def cw_attack(sess, dataset='MNIST', ensemble_size=1,
              model_dir='models/default', confidence=0,
              num_samples=10000):
    with tf.Session() as sess:
        if dataset == 'MNIST':
            data, model =  MNIST(), MNISTModel(session=sess,
                                               ensemble_size=ensemble_size,
                                               model_dir=model_dir)
        else:
            data, model =  CIFAR(), CIFARModel(session=sess,
                                               ensemble_size=ensemble_size,
                                               model_dir=model_dir)

        attack = CarliniL2(sess, model, batch_size=200, max_iterations=1000,
                           confidence=confidence, targeted=False)

        inputs, targets = generate_data(data, samples=num_samples, targeted=False,
                                        start=0, inception=False)

        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")


        return np.squeeze(adv)
