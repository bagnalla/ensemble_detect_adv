import pickle, gzip, random, os.path
from functools import reduce
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 32
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
FC_SIZE_1 = 384
FC_SIZE_2 = 192
WEIGHT_DECAY = 0.004
AGREE_PEN = 1.0
LAMBDA = 1

def confidence_sums(logits):
    confs = [confidences(l) for l in logits]
    return tf.reduce_sum(confs, axis=0)

def _variable_on_cpu(name, shape, initializer):
  dtype = tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float32
  var = tf.get_variable(
      name, shape,
      initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
      dtype=dtype)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images, name='m0', reuse=None):
    batch_size = int(images.shape[0])

    # conv1
    with tf.variable_scope(name + '_conv1', reuse=reuse) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64],
                                  tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope(name + '_conv2', reuse=reuse) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME',
                               name='pool2')

    # local3
    with tf.variable_scope(name + '_local3', reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, FC_SIZE_1],
                                              stddev=0.04, wd=WEIGHT_DECAY)
        biases = _variable_on_cpu('biases', [FC_SIZE_1],
                                  tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)

    # local4
    with tf.variable_scope(name + '_local4', reuse=reuse) as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape=[FC_SIZE_1, FC_SIZE_2],
                                              stddev=0.04, wd=WEIGHT_DECAY)
        biases = _variable_on_cpu('biases', [FC_SIZE_2],
                                  tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,
                            name=scope.name)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope(name + '_softmax_linear', reuse=reuse) as scope:
        weights = _variable_with_weight_decay('weights',
                                              [FC_SIZE_2, NUM_CLASSES],
                                              stddev=1.0/FC_SIZE_2, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases,
                                name=scope.name)

    return softmax_linear

def loss(logits, y, n):
    logits_train = [l[:n,:] for l in logits]
    logits_adv = [l[n:,:] for l in logits]
    y_train = y[:n]

    labels = tf.cast(y_train, tf.int64)
    cross_entropy = [tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=l, name='cross_entropy_per_example')
                     for l in logits_train]
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    weight_reg = tf.add_n(tf.get_collection('losses')) / len(logits)

    # The loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    loss = cross_entropy_mean + weight_reg

    # Agree penalty term
    agree_penalties = []
    confs = [confidences(l) for l in logits_adv]
    for i in range(len(confs)):
        for j in range(len(confs)):
            if j <= i: continue
            pen = tf.reduce_mean(tf.square(tf.reduce_sum(
                tf.multiply(confs[i], confs[j]), axis=1)))
            agree_penalties.append(pen)
    m = len(agree_penalties)
    agree_penalty = (LAMBDA / m if m > 0 else 1.0) * sum(agree_penalties)

    return loss + agree_penalty

# Used to generate adv examples against our ensembles
def adv_loss(logits, y, lam=LAMBDA):
    labels = tf.cast(y, tf.int64)
    cross_entropy = [tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=l, name='cross_entropy_per_example')
                     for l in logits]
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    weight_reg = tf.add_n(tf.get_collection('losses')) / len(logits)

    # The loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    loss = cross_entropy_mean + weight_reg

    # Agree penalty term
    agree_penalties = []
    confs = [confidences(l) for l in logits]
    for i in range(len(confs)):
        for j in range(len(confs)):
            if j <= i: continue
            pen = tf.reduce_mean(tf.square(tf.reduce_sum(
                tf.multiply(confs[i], confs[j]), axis=1)))
            agree_penalties.append(pen)
    m = len(agree_penalties)
    agree_penalty = (AGREE_PEN / m if m > 0 else 1.0) * sum(agree_penalties)

    return loss + lam * agree_penalty

def training(loss, x, learning_rate, decay_step, decay_factor):
  global_step = tf.Variable(0, name='global_step', trainable=False)
  lr = tf.train.exponential_decay(learning_rate, global_step, decay_step,
                                  decay_factor, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(lr)
  grads = optimizer.compute_gradients(loss)
  train_op = optimizer.apply_gradients(grads)
  return train_op

def evaluation(logits, labels):
  return tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.int32))

def predictions(logits):
  return tf.cast(tf.argmax(logits, axis=1), tf.int32)

def ranks(logits):
  return tf.nn.top_k(logits, NUM_CLASSES, sorted=True).indices

def confidences(logits):
    return tf.nn.softmax(logits)

def predictions_confidences(logits):
  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
  confs = confidences(logits)
  confs = tf.reduce_sum(confs * tf.one_hot(predictions, NUM_CLASSES), axis=1)
  return predictions, confs

def save_weights(sess, dir = 'models'):
  os.makedirs(dir, exist_ok=True)
  all_vars = tf.trainable_variables()
  with gzip.open(dir + "/cifar10_params.pkl.gz", "w") as f:
    pickle.dump(tuple(map(lambda x: x.eval(sess), all_vars)), f)

def load_weights(sess, model_names, dir):
  i = 0
  filename = dir + '/cifar10_params.pkl.gz' if dir else 'cifar10_params.pkl.gz'
  with gzip.open(filename, 'rb') as f:
    weights = pickle.load(f, encoding='latin1')
  for name in model_names:
    c1w, c1b, c2w, c2b, l3w, l3b, l4w, l4b, smw, smb = tuple(weights[i:i+10])
    i += 10
    with tf.variable_scope(name + '_conv1', reuse=True):
      w_var = tf.get_variable('weights', [5, 5, 3, 64])
      b_var = tf.get_variable("biases", [64])
      sess.run(w_var.assign(c1w))
      sess.run(b_var.assign(c1b))
    with tf.variable_scope(name + '_conv2', reuse=True):
      w_var = tf.get_variable('weights', [5, 5, 64, 64])
      b_var = tf.get_variable("biases", [64])
      sess.run(w_var.assign(c2w))
      sess.run(b_var.assign(c2b))
    with tf.variable_scope(name + '_local3', reuse=True):
      w_var = tf.get_variable('weights', [4096, FC_SIZE_1])
      b_var = tf.get_variable("biases", [FC_SIZE_1])
      sess.run(w_var.assign(l3w))
      sess.run(b_var.assign(l3b))
    with tf.variable_scope(name + '_local4', reuse=True):
      w_var = tf.get_variable('weights', [FC_SIZE_1, FC_SIZE_2])
      b_var = tf.get_variable("biases", [FC_SIZE_2])
      sess.run(w_var.assign(l4w))
      sess.run(b_var.assign(l4b))
    with tf.variable_scope(name + '_softmax_linear', reuse=True):
      w_var = tf.get_variable('weights', [FC_SIZE_2, 10])
      b_var = tf.get_variable("biases", [10])
      sess.run(w_var.assign(smw))
      sess.run(b_var.assign(smb))
