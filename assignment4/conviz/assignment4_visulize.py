import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

PLOT_DIR = './out/plots'

def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

def main():



    with tf.Session() as sess:

        saver = tf.train.import_meta_graph('/home/thtran/workspace/assignment4/tmp/cifar10_train/model.ckpt-100000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('/home/thtran/workspace/assignment4/tmp/cifar10_train/'))

        conv_weights = sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv4/weights')])
		
        # for i, c in enumerate(conv_weights[0]):
            # np.save('./khanh' + str(i), c)

        for i, c in enumerate(conv_weights[0]):
			plot_conv_weights(c, 'cifar_conv{}'.format(i))
		#np.save('./lam' + str(i), x)

  

if(__name__ == "__main__"):
    main()
