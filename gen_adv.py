import argparse, os.path, pickle, sys
import numpy as np
import tensorflow as tf

from adv_lib import basic_set, DF_set, FGS_op, gen_FGS_set, random_perturbation
from cw_attack import cw_attack
from dataset_params import choose_dataset

FLAGS = None
FGS, BI, DF, CW, RAND = 0, 1, 2, 3, 4


def gen():
    # Load parameters and data for the chosen dataset.
    model, save_images, NUM_CLASSES, IMAGE_SIZE, example_shape, load_data \
        = choose_dataset(FLAGS.dataset)
    _, validation_data, test_data = load_data()
    if FLAGS.subset == 0:
        images, labels = test_data.images, test_data.labels
    else:
        images, labels = validation_data.images, validation_data.labels

    model_names = ['m' + str(i) for i in range(FLAGS.ensemble_size)]

    with tf.Graph().as_default():
        if FLAGS.attack in [FGS, CW]:
            x = tf.placeholder(tf.float32,
                               shape=example_shape(FLAGS.batch_size))
            y = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        elif FLAGS.attack in [BI, DF]:
            x = tf.placeholder(tf.float32, shape=example_shape(1))
            y = tf.placeholder(tf.int32, shape=(1))
        elif FLAGS.attack != RAND:
            print('Unrecognized attack type.')
            return

        if FLAGS.attack not in [CW, RAND]:
            logits = [model.inference(x, name) for name in model_names]
            loss = model.adv_loss(logits, y, lam=FLAGS.lam)
        if FLAGS.attack == DF:
            sum_logits = tf.reduce_sum(logits, axis=0)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if FLAGS.attack not in [CW, RAND]:
            model.load_weights(sess, model_names, FLAGS.model_dir)

        # Fast gradient sign
        if FLAGS.attack == FGS:
            fgs_op = FGS_op(loss, x, FLAGS.epsilon)
            adv_examples = gen_FGS_set(sess, x, y, fgs_op, images, labels,
                                       FLAGS.batch_size)

        # Basic iterative
        elif FLAGS.attack == BI:
            adv_examples = basic_set(sess, x, y, FLAGS.epsilon, loss,
                                     images, labels, num_iters=10)

        # DeepFool
        elif FLAGS.attack == DF:
            pred_op = model.predictions(sum_logits)
            grad_ops = [tf.gradients(sum_logits[:, i], x)
                        for i in range(int(sum_logits.get_shape()[1]))]
            adv_examples, perturbations = DF_set(
                sess, x, y, sum_logits, pred_op, grad_ops, images, NUM_CLASSES)

        # C&W l2
        elif FLAGS.attack == CW:
            adv_examples = cw_attack(sess, dataset=FLAGS.dataset,
                                     ensemble_size=FLAGS.ensemble_size,
                                     model_dir=FLAGS.model_dir,
                                     confidence=FLAGS.cw_confidence,
                                     num_samples=images.shape[0])

        # Random noise
        elif FLAGS.attack == RAND:
            adv_examples = images + random_perturbation(images.shape,
                                                        FLAGS.epsilon,)
        else:
            print('--type argument not recognized.')
            return

        if FLAGS.direct == 1:
            dir = 'adv_examples/'
        else:
            dir = 'adv_examples/' + str(FLAGS.attack) + '_' + str(FLAGS.epsilon)
        os.makedirs(dir, exist_ok=True)

        # The attacks should produce clipped examples already, but we
        # make sure they are clipped here.
        adv_examples = np.clip(adv_examples, 0, 1)

        print('Writing adversarial examples to disk...')
        save_images(adv_examples[:9], dir)
        with open(dir + '/adv.pkl.gz', 'wb') as f:
            pickle.dump(np.squeeze(adv_examples), f)


def main(_):
    gen()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '-e', '--epsilon',
        type=float,
        default=0.03,
        help='Epsilon value for FGS and Basic iterative.'
    )
    parser.add_argument(
        '-n', '--ensemble_size',
        type=int,
        default=1,
        help='Number of ensemble models.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models/default',
        help='Directory to load the weights from.'
    )
    parser.add_argument(
        '-a', '--attack',
        type=int,
        default=0,
        help='0) FGS. 1) Basic iterative. 2) DeepFool. 3) C&W l2. 4) random noise.'
    )
    parser.add_argument(
        '--direct',
        type=int,
        default=0,
        help='1 to save directly to adv_examples directory.'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='MNIST',
        help='Dataset (either MNIST or CIFAR-10).'
    )
    parser.add_argument(
        '-s', '--subset',
        type=int,
        default=0,
        help='Subset of the dataset to use for testing. 0) test  1) validation'
    )
    parser.add_argument(
        '-c', '--cw_confidence',
        type=float,
        default=0.0,
        help='Confidence parameter of the C&W l2 attack.'
    )
    parser.add_argument(
        '-l', '--lam',
        type=float,
        default=0.0,
        help='Lambda parameter of the FGS and basic iterative attacks.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
