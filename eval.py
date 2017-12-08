import argparse, sys, pickle
import numpy as np
import tensorflow as tf
from dataset_params import choose_dataset
from util import run_batches

FLAGS = None


def load_adv_examples():
    with open(FLAGS.adv_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def init_placeholders(batch_size, example_shape):
    x = tf.placeholder(tf.float32, example_shape)
    y = tf.placeholder(tf.int32, shape=(batch_size))
    return x, y


def init_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess


def eval_with_rejection(sess, x, sum_logits_op, rank_ops, images, labels):
    logit_sums = run_batches(sess, sum_logits_op, [x], [images], FLAGS.batch_size)
    preds = np.argmax(logit_sums, axis=1)

    print(labels[0:20])
    print(preds[0:20])

    num_correct_raw = np.sum(preds == labels)
    acc = num_correct_raw / images.shape[0]
    print('Ensemble accuracy = %.4f' % acc)

    all_ranks = []
    for j in range(len(rank_ops)):
        rank_op = rank_ops[j]
        ranks = np.array(run_batches(sess, rank_op, [x], [images],
                                     FLAGS.batch_size))
        all_ranks.append(ranks)

    all_ranks = np.array(all_ranks).T
    rank_sum_correct, rank_sum_wrong = 0, 0
    num_correct, num_accepted, num_detected = 0, 0, 0
    predictions = []
    num_wrong, num_detected_and_wrong, num_detected_and_correct = 0, 0, 0
    # For each image
    for j in range(all_ranks.shape[1]):
        ranks = all_ranks[:, j, :]
        sums = np.zeros([10])
        # For each model
        for k in range(ranks.shape[1]):
            model_ranks = np.squeeze(ranks[:, k])
            # For each label
            for i in range(model_ranks.shape[0]):
                lbl = model_ranks[i]
                sums[lbl] += i
        pred = np.argmin(sums)
        pred_rank = sums[pred]

        wrong = False

        # use sum_logit predictions
        if preds[j] == labels[j]:
            rank_sum_correct += pred_rank
        else:
            rank_sum_wrong += pred_rank
            wrong = True
            num_wrong += 1

        if pred_rank <= FLAGS.rank_threshold:
            num_accepted += 1
            predictions.append(pred)
            if not wrong:
                num_correct += 1
        else:
            num_detected += 1
            if wrong:
                num_detected_and_wrong += 1
            else:
                num_detected_and_correct += 1

    print("accepted: %0.04f" % (num_accepted / all_ranks.shape[1]))
    print("accuracy on accepted: %0.04f" % (num_correct / num_accepted
                                            if num_accepted != 0 else 1.0))
    print("accuracy overall: %0.04f" % (num_correct / all_ranks.shape[1]))


def eval_adv(model, x, y, rank_ops, sum_logits_op, model_names, images, labels):
    sess = init_session()
    adv_examples = load_adv_examples()

    model.load_weights(sess, model_names, FLAGS.model_dir)

    print('\nEvaluating ensemble on original test examples.')
    eval_with_rejection(sess, x, sum_logits_op, rank_ops, images, labels)

    labels = labels[:adv_examples.shape[0]]

    print('\nEvaluating ensemble on adversarial test examples.')
    eval_with_rejection(sess, x, sum_logits_op, rank_ops, adv_examples, labels)


def main(argv):
    model, save_images, NUM_CLASSES, IMAGE_SIZE, example_shape, load_data \
        = choose_dataset(FLAGS.dataset)
    _, validation_data, test_data = load_data()
    images = test_data.images if FLAGS.set == 0 else validation_data.images
    labels = test_data.labels if FLAGS.set == 0 else validation_data.labels

    with tf.Graph().as_default():
        print("building computation graph...")
        x, y = init_placeholders(FLAGS.batch_size,
                                 example_shape(FLAGS.batch_size))
        model_names = ['m' + str(i) for i in range(FLAGS.ensemble_size)]
        logits = [model.inference(x, name=nm) for nm in model_names]
        rank_ops = [model.ranks(l) for l in logits]
        sum_logits = tf.reduce_sum(logits, axis=0)

        eval_adv(model, x, y, rank_ops, sum_logits, model_names, images,
                 labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=200,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='../../mnist/data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='models/default',
        help='Directory to load the weights from.'
    )
    parser.add_argument(
        '--adv_path',
        type=str,
        default='adv_examples/adv.pkl.gz',
        help='Location of the adversarial examples.'
    )
    parser.add_argument(
        '--ensemble_size', '-n',
        type=int,
        default=1,
        help='.'
    )
    parser.add_argument(
        '--rank_threshold', '-rt',
        type=int,
        default=0,
        help='Rank threshold for rejection.'
    )
    parser.add_argument(
        '--experiment', '-ex',
        type=str,
        default='default_experiment',
        help='Experiment name.'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='MNIST',
        help='Dataset (either MNIST or CIFAR-10).'
    )
    parser.add_argument(
        '-s', '--set',
        type=int,
        default=0,
        help='0 to evaluate on test set, 1 for validation set.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
