
# import
import os
import sys
import time
import copy
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from tf_cnnvis import *

# tensorflow model implementation (Alexnet convolution)
X = tf.placeholder(tf.float32, shape = [None, 24, 24, 3]) # placeholder for input images
y_ = tf.placeholder(tf.float32, shape = [None, 10]) # placeholder for true labels for input images



# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver = tf.train.import_meta_graph('/home/thtran/workspace/assignment4/tmp/cifar10_train/model.ckpt-100000.meta')
  saver.restore(sess, '/home/thtran/workspace/assignment4/tmp/cifar10_train/model.ckpt-100000')
  print("Model restored.")
  
  # reading sample image
  im = imresize(imread(os.path.join("./sample_images", "images.jpg")), (24, 24))
  im = tf.image.per_image_standardization(im)
  im = np.expand_dims(im.eval(), axis = 0)
  # activation visualization
  layers = ['conv1/weights']
  for op in tf.get_default_graph().get_operations():
      print(op.name) 
      
  start = time.time()
  is_success = activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {X : im}, 
                                      layers=layers, path_logdir="./Log", path_outdir="./Output")
  start = time.time() - start
  print("Total Time = %f" % (start))

  # deconv visualization
  layers = ['conv1/weights']

  start = time.time()
  is_success = deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = {X : im}, 
                                  layers=layers, path_logdir="./Log", path_outdir="./Output")
  start = time.time() - start
  print("Total Time = %f" % (start))