import tensorflow as tf
from castanea.initializers import xavier_initializer_conv2d, xavier_initializer_conv2d_transpose
from castanea.normalize import *
from castanea.layers.parameter import LayerParameter
from castanea.utils import device_or_none

def bn_beta_gamma(target, beta_name=None, gamma_name=None, var_device=None):
    """create beta,gamma variable for batch_normalization"""
    with device_or_none(var_device):
        return (
            tf.get_variable(beta_name, shape=target.get_shape().as_list()[-1],
                initializer=tf.zeros_initializer()), 
            tf.get_variable(gamma_name, shape=target.get_shape().as_list()[-1],
                initializer=tf.ones_initializer()) )

def conv2d(
        x, kernel_height, kernel_width, out_channels,
        strides=[1,1,1,1], parameter=None):

    paramter = parameter or LayerParameter()

    with tf.variable_scope(None, default_name='conv2d', reuse=None):
        x_shape = x.get_shape().as_list()

        with device_or_none(parameter.var_device):
            weight = tf.get_variable(
                shape=[kernel_height, kernel_width, x_shape[3], out_channels],
                initializer=xavier_initializer_conv2d(x, kernel_height, kernel_width),
                name='weight')

        if parameter.with_weight_normalize:
            weight = normalize_weight_for_conv2d(weight, var_device=parameter.var_device)

        out = tf.nn.conv2d(
            x, weight, strides=strides, padding=parameter.padding)

        if paramter.with_bias:
            with device_or_none(parameter.var_device):
                bias = tf.get_variable(
                    shape=[out_channels], initializer=tf.zeros_initializer(), name='bias')

            out = tf.nn.bias_add(out, bias)

        if paramter.with_batch_normalize:
            mean, var = tf.nn.moments(out, [0,1,2])
            beta, gamma = bn_beta_gamma(out,'beta', 'gamma', parameter.var_device)
            out = tf.nn.batch_normalization(out, mean, var, beta, gamma, 1e-6)

        if parameter.rectifier:
            out = parameter.rectifier(out)

        return out

def conv2d_transpose(
        x, kernel_height, kernel_width, out_channels,
        strides=[1,2,2,1], parameter=None):

    paramter = parameter or LayerParameter()

    with tf.variable_scope(None, default_name='conv2d_transpose'):
        x_shape = x.get_shape().as_list()

        with device_or_none(parameter.var_device):
            weight = tf.get_variable(
                shape=[kernel_height, kernel_width, out_channels, x_shape[3]],
                initializer=xavier_initializer_conv2d_transpose(x, kernel_height, kernel_width),
                name='weight')

        if parameter.with_weight_normalize:
            weight = normalize_weight_for_conv2d_transpose(weight, var_device=parameter.var_device)

        out = tf.nn.conv2d_transpose(
            x, weight, [x_shape[0], x_shape[1] * strides[1], x_shape[2] * strides[2], out_channels],
            strides=strides, padding=parameter.padding)

        if paramter.with_bias:
            with device_or_none(parameter.var_device):
                bias = tf.get_variable(
                    shape=[out_channels], initializer=tf.zeros_initializer(), name='bias')

            out = tf.nn.bias_add(out, bias)

        if paramter.with_batch_normalize:
            mean, var = tf.nn.moments(out, [0,1,2])
            beta, gamma = bn_beta_gamma(out,'beta', 'gamma', parameter.var_device)
            out = tf.nn.batch_normalization(out, mean, var, beta, gamma, 1e-6)

        if parameter.rectifier:
            out = parameter.rectifier(out)

        return out

def separable_conv2d(
        x, kernel_height, kernel_width, channel_multiplier, out_channels,
        strides=[1,1,1,1], parameter=None):

    paramter = parameter or LayerParameter()

    with tf.variable_scope(None, default_name='separable_conv2d'):
        x_shape = x.get_shape().as_list()

        with device_or_none(parameter.var_device):
            depthwise_weight = tf.get_variable(
                shape=[kernel_height, kernel_width, x_shape[3], channel_multiplier],
                initializer=xavier_initializer_conv2d(x, kernel_height, kernel_width),
                name='depthwise_weight')
            pointwise_weight = tf.get_variable(
                shape=[1, 1, shale[3] * channel_multiplier, out_channels],
                initializer=tf.ones_initializer(),
                name='pointwise_weight')

        out = tf.nn.separable_conv2d(
            x, depthwise_weight, pointwise_weight, strides, padding=parameter.padding)

        if paramter.with_bias:
            with device_or_none(parameter.var_device):
                bias = tf.get_variable(
                    shape=[out_channels], initializer=tf.zeros_initializer(), name='bias')

            out = tf.nn.bias_add(out, bias)

        if parameter.rectifier:
            out = parameter.rectifier(out)

        return out

