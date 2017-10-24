import tensorflow as tf
import numpy as np
import castanea.models.downloader as downloader

VGG16_NAME='vgg16'
VGG16MAT_FILENAME='imagenet-vgg-verydeep-16.mat'
VGG16MAT_URL='http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat'

vgg_mat = None

def get_mat(filepath=None):
    filepath = filepath or VGG16MAT_FILENAME
    return downloader.download_mat_file(filepath, VGG16MAT_URL)

def _inference(image, vgg_mat, with_fc, training, trainable, reuse, name):
    vgg_mat = np.squeeze(vgg_mat['layers'])
    x = image
    d = {}

    with tf.variable_scope(name, reuse=reuse):
        for i in range(len(vgg_mat)):
            layer_name = vgg_mat[i][0][0][0][0]
            layer_type = vgg_mat[i][0][0][1][0]

            if layer_type == 'conv':
                kernel, bias = vgg_mat[i][0][0][2][0]
                kernel = np.transpose(kernel, (1,0,2,3))

                init = tf.constant_initializer(kernel, dtype=tf.float32)
                w = tf.get_variable(
                    name=layer_name+'_w', initializer=init,
                    shape=kernel.shape, trainable=trainable)
                x = tf.nn.conv2d(x, w, [1,1,1,1], padding='SAME')

                bias = bias.reshape(-1)
                init = tf.constant_initializer(bias, dtype=tf.float32)
                b = tf.get_variable(
                    name=layer_name+'_b', initializer=init,
                    shape=bias.shape, trainable=trainable)

                d[layer_name] = x = tf.nn.bias_add(x, b)
            elif layer_type == 'relu':
                d[layer_name] = x= tf.nn.relu(x)
            elif layer_type == 'pool':
                pool_type = vgg_mat[i][0][0][2][0]
                if pool_type == 'max':
                    d[layer_name] = x = tf.nn.max_pool(
                        x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

            if layer_name == 'pool5':
                break

    return x, d

def inference(image, vgg_mat, training, trainable=False, reuse=False, name=VGG16_NAME):
    x, d = _inference(image, vgg_mat, False, training, trainable, reuse, name)
    return x

def inference_as_dict(image, vgg_mat, training, trainable, reuse=False, name=VGG16_NAME):
    x, d = _inference(image, vgg_mat, False, training, trainable, reuse, name)
    return d

