from random import random
import time
import numpy as np
import tensorflow as tf


def FGS_op(loss, x, eps):
    x_grad = tf.gradients(loss, x)
    sign_grad = tf.sign(x_grad)
    return x + eps * sign_grad


def FGS_randomized_op(loss, x, eps):
    x_grad = tf.gradients(loss, x)
    sign_grad = tf.sign(x_grad) * tf.random_uniform(tf.shape(x_grad),
                                                    minval=0.0, maxval=1.0)
    return x + eps * sign_grad


def gen_FGS_set(sess, x, y, gen_op, images, labels, batch_size):
    i = 0
    adv_images = np.empty((0, 784))
    while i < images.shape[0]:        
        next_i = min(i + batch_size, images.shape[0])
        feed_dict = { x: images[i:next_i,:], y: labels[i:next_i] }
        adv_images = np.append(adv_images, sess.run([gen_op],
                                                    feed_dict=feed_dict))
        i = next_i
    return adv_images.reshape(images.shape)


def gen_rand_FGS_set(sess, x, y, fgs_op, images, labels, epsilon, batch_size):
    perturbed_images = images + random_perturbation(images.shape, epsilon)
    return gen_FGS_set(sess, x, y, fgs_op, perturbed_images, labels, batch_size)


# Full iterative DeepFool algorithm for an entire set of images
def DF_set(sess, x, y, logits, pred_op, grad_ops, images, num_classes,
           display=False, max_iters=50):
    final_images, final_perturbations = [], []
    for j in range(images.shape[0]):
        if j % 100 == 0: print(j)
        if display and j % 1000 == 0: print('DF %d' % j)
        i = 0
        xs = [images[j][np.newaxis,:]] # versions of the image as we perturb.
                                       # first is the original
        rs = [] # perturbations
        original_pred = sess.run([pred_op], feed_dict={ x: xs[0] })[0]
        new_pred = original_pred
        grad_pred = sess.run([grad_ops[original_pred[0]]],
                             feed_dict={ x: xs[i] })[0][0]
        while new_pred == original_pred:
            w_ks, f_ks = [], []
            for k in range(num_classes):
                if k == original_pred: continue

                grad_k = sess.run([grad_ops[k]], feed_dict={ x: xs[i] })[0][0]
                w_k_prime = grad_k - grad_pred
                w_ks.append(w_k_prime)

                logits_value = sess.run([logits], feed_dict={ x: xs[i] })[0][0]
                f_k = logits_value[k]
                f_pred = logits_value[original_pred[0]]

                f_k_prime = f_k - f_pred
                f_ks.append(f_k_prime)
            
            l = np.array(list(map(lambda k: np.abs(f_ks[k]) /
                                  np.linalg.norm(w_ks[k]), range(len(w_ks)))))
            l_hat = np.argmin(l)
            if (f_ks[l_hat] == 0.0):
                break

            rs.append((np.abs(f_ks[l_hat]) /
                       np.sum(np.square(w_ks[l_hat]))) * w_ks[l_hat])
            xs.append(np.clip(xs[i] + rs[i], 0, 1))

            i += 1
            new_pred = sess.run([pred_op], feed_dict={ x: xs[i] })[0]
            if i > max_iters:
                # print('reached max iters')
                break

        final_perturbations.append(sum(rs))
        final_images.append(images[j] + final_perturbations[j] * 1.02)

    return np.squeeze(np.array(final_images)), np.array(final_perturbations)


# One-shot DF for a single image
def DF_fast(image, label, logits, grads, num_classes):
    original_pred = np.argmax(logits)
    
    if original_pred != label:
        return image, np.zeros(image.shape)
    
    grad_pred = grads[original_pred]
    w_ks = []
    w_k_norms = []
    f_ks = []

    for k in range(num_classes):
        if k == original_pred: continue
            
        grad_k = grads[k]
        w_k_prime = grad_k - grad_pred
        w_k_norm = np.linalg.norm(w_k_prime)
        if w_k_norm == 0.0: continue
        w_k_norms.append(w_k_norm)
        w_ks.append(w_k_prime)
        
        f_k = logits[k]
        f_pred = logits[original_pred]
        
        f_k_prime = f_k - f_pred
        f_ks.append(f_k_prime)
        
    l = np.array(list(map(lambda k: np.abs(f_ks[k]) /
                          w_k_norms[k], range(len(w_ks)))))
    l_hat = np.argmin(l)

    r = (np.abs(f_ks[l_hat]) / np.sum(np.square(w_ks[l_hat]))) * \
        w_ks[l_hat] * 1.1
    
    return image + r, r


# One-shot DF for a single image (target a specific label)
def DF_fast_target(image, logits, grads, target, num_classes):
    original_pred = np.argmax(logits)
    if original_pred == target: return image, np.zeros(image.shape)

    grad_pred = grads[original_pred]
    w_ks = []
    f_ks = []

    for k in range(num_classes):
        grad_k = grads[k]
        w_k_prime = grad_k - grad_pred
        w_ks.append(w_k_prime)
        
        f_k = logits[k]
        f_pred = logits[original_pred]
        
        f_k_prime = f_k - f_pred
        f_ks.append(f_k_prime)
    
    l_hat = target

    r = (np.abs(f_ks[l_hat]) / np.sum(np.square(w_ks[l_hat]))) * \
        w_ks[l_hat] * 1.1
    
    return image + r, r


# Given actual values for logits and grads, produce DF_fast examples
def gen_DF_fast(images, labels, logits, grads, num_classes):
    perturbed_images = []
    for i in range(images.shape[0]):
        perturbed, _ = DF_fast(images[i], labels[i], logits[i], grads[i],
                               num_classes)
        perturbed_images.append(perturbed)
    return np.squeeze(np.array(perturbed_images))


# Generate DF_fast set given the necessary ops
def DF_fast_set(sess, x, y, logits, grad_op, images, labels, batch_size,
                num_classes):
    i, adv_images = 0, np.empty((0, 784))
    while i < images.shape[0]:
        next_i = min(i + batch_size, images.shape[0])
        feed_dict = { x: images[i:next_i,:] }
        logits_values = sess.run(logits, feed_dict=feed_dict)
        grad_values = sess.run(grad_op, feed_dict=feed_dict)
        grad_values = np.swapaxes(np.squeeze(np.array(grad_values)), 0, 1)
        adv_images = np.append(adv_images, gen_DF_fast(
            images[i:next_i,:], labels[i:next_i], logits_values, grad_values,
            num_classes))
        i = next_i
    return adv_images.reshape(images.shape)


# Generate DF_fast set given the necessary ops
# Add random perturbations before DF
def rand_DF_fast_set(sess, x, y, logits, grad_ops, images, labels, epsilon,
                     batch_size, num_classes):
    perturbed_images = images + random_perturbation(images.shape, epsilon)
    return DF_fast_set(sess, x, y, logits, grad_ops, perturbed_images, labels,
                       batch_size, num_classes)


# Generate DF_full set given the necessary ops
# Add random perturbations before DF
def rand_DF_set(sess, x, y, logits, pred_op, grad_ops, images, epsilon,
                num_classes, display=False):
    perturbed_images = images + random_perturbation(images.shape, epsilon)
    return DF_set(sess, x, y, logits, pred_op, grad_ops, perturbed_images,
                  display, num_classes)


def random_perturbation(shape, epsilon):
    return (np.random.random_integers(0, 1, size=shape) * 2 - 1) * \
        np.random.random(shape) * epsilon


# Not used currently
from functools import reduce
def random_l2_perturbation(shape, delta):
    n = reduce(lambda x, y: x * y, shape)
    a = np.random.random_sample(n) * 2 - 1
    a = a.reshape(shape[0], -1)
    a = a / np.linalg.norm(a, axis=1)[:,np.newaxis]
    a = a.reshape(shape)
    return a * delta


# Basic iterative method for a full set of images
def basic_set(sess, x, y, epsilon, loss_op, images, labels, num_iters=10):
    adv_images = []
    x_grad_op = tf.gradients(loss_op, x)
    sign_grad_op = (epsilon / num_iters) * tf.sign(x_grad_op)

    start_time = time.time()
    for i in range(images.shape[0]):
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print("%d :: %.02f" % (i, elapsed_time))
        original_image = images[i][np.newaxis,:]
        feed_dict = { x: original_image, y: labels[np.newaxis,i] }
        total_perturb = np.zeros(images[i].shape)
        for j in range(num_iters):
            perturb = np.squeeze(sess.run(sign_grad_op, feed_dict=feed_dict))
            total_perturb = np.clip(total_perturb + perturb, -epsilon, epsilon)
            total_perturb = np.clip(total_perturb, -original_image, 1 - original_image)
            feed_dict[x] = original_image + total_perturb
        adv_images.append(np.squeeze(feed_dict[x]))

    return np.array(adv_images)
