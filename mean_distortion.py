import argparse, sys, time, pickle, os.path
from six.moves import xrange
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from dataset_params import choose_dataset

def load_adv():
    with open(FLAGS.adv, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def main(argv):
    model, save_images, NUM_CLASSES, IMAGE_SIZE, example_shape, load_data \
        = choose_dataset(FLAGS.dataset)
    train_data, validation_data, test_data = load_data()
    adv_images = load_adv()
    clean_images = test_data.images[:adv_images.shape[0]]

    diff = clean_images - adv_images
    diff = diff.reshape([diff.shape[0], -1])
    norms = np.linalg.norm(diff, axis=1)
    avg_norm = np.mean(norms)
    
    print('mean distortion: %.04f' % avg_norm)

    # Sanity check
    # distortion_sum = 0
    # for i in range(len(adv_images)):
    #     distortion_sum += np.sum((adv_images[i]-clean_images[i])**2)**.5
    # print('mean distortion 2: ' + str(distortion_sum / len(adv_images)))

    # Save images
    # save_images(clean_images[:20], '.', 'clean.jpg')
    # save_images(adv_images[:20], '.', 'adv.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='mnist/data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--adv',
        type=str,
        default='adv_examples/adv.pkl.gz',
        help='Directory to save the weights.'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='MNIST',
        help='Dataset (either MNIST or CIFAR-10).'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
