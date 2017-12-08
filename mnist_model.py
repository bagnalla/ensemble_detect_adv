import gzip, pickle, os.path
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_SIZE = 128
WEIGHT_DECAY = 0.00002
LAMBDA = 1.0


def confidence_sums(logits):
    confs = [confidences(l) for l in logits]
    return tf.reduce_sum(confs, axis=0)


def inference(images, name='m0', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        w1 = tf.get_variable("w1", (IMAGE_PIXELS, HIDDEN_SIZE),
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [HIDDEN_SIZE],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2", (HIDDEN_SIZE, NUM_CLASSES),
                             initializer=tf.contrib.layers.xavier_initializer())
        l1 = tf.nn.relu(tf.matmul(images, w1) + b1)
        l2 = tf.matmul(l1, w2)
        # add weight decay to 'losses" collection
        weight_decay = tf.multiply(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2),
                                   WEIGHT_DECAY, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return l2


def loss(logits, y, n):
    logits_train = [l[:n, :] for l in logits]
    logits_adv = [l[n:, :] for l in logits]
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
            if j <= i:
                continue
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
            if j <= i:
                continue
            pen = tf.reduce_mean(tf.square(tf.reduce_sum(
                tf.multiply(confs[i], confs[j]), axis=1)))
            agree_penalties.append(pen)
    m = len(agree_penalties)
    agree_penalty = (LAMBDA / m if m > 0 else 1.0) * sum(agree_penalties)

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


def save_weights(sess, dir='models'):
    os.makedirs(dir, exist_ok=True)
    all_vars = tf.trainable_variables()
    with gzip.open(dir + "/mnist_params.pkl.gz", "w") as f:
        pickle.dump(tuple(map(lambda x: x.eval(sess), all_vars)), f)


def load_weights(sess, model_names, dir):
    i = 0
    filename = dir + '/mnist_params.pkl.gz' if dir else 'mnist_params.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')
    for name in model_names:
            w1, b1, w2 = tuple(weights[i:i+3])
            i += 3
            with tf.variable_scope(name, reuse=True):
                w1_var = tf.get_variable("w1", (IMAGE_PIXELS, HIDDEN_SIZE))
                b1_var = tf.get_variable("b1", (HIDDEN_SIZE))
                w2_var = tf.get_variable("w2", (HIDDEN_SIZE, NUM_CLASSES))
                sess.run([w1_var.assign(w1), b1_var.assign(b1),
                          w2_var.assign(w2)])
