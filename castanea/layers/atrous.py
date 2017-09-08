import tensorflow as tf
from castanea.initializers import xavier_initializer_conv2d, xavier_initializer_conv2d_transpose
from castanea.normalize import *
from castanea.layers.parameter import LayerParameter
from castanea.utils import device_or_none

def atrous_conv2d(x, kernel_height, kernel_width, rate, out_channels, parameter=None):

    paramter = parameter or LayerParameter()
    var_scope_default_name = parameter.var_scope_default_name or 'atrous_conv2d'

    with tf.variable_scope(None, default_name=var_scope_default_name, reuse=None):
        x_shape = x.get_shape().as_list()

        with device_or_none(parameter.var_device):
            weight = tf.get_variable(
                shape=[kernel_height, kernel_width, x_shape[3], out_channels],
                initializer=xavier_initializer_conv2d(x, kernel_height, kernel_width),
                name='weight')

        if parameter.with_weight_normalize:
            weight = normalize_weight_for_conv2d(weight, var_device=parameter.var_device)

        out = tf.nn.atrous_conv2d(x, weight, rate, padding=parameter.padding)

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

def atrous_conv2d_transpose(
        x, kernel_height, kernel_width, rate, out_channels, parameter=None):

    paramter = parameter or LayerParameter()
    var_scope_default_name = parameter.var_scope_default_name or 'atrous_conv2d_transpose'

    with tf.variable_scope(None, default_name=var_scope_default_name, reuse=None):
        x_shape = x.get_shape().as_list()

        with device_or_none(parameter.var_device):
            weight = tf.get_variable(
                shape=[kernel_height, kernel_width, out_channels, x_shape[3]],
                initializer=xavier_initializer_conv2d_transpose(x, kernel_height, kernel_width),
                name='weight')

        if parameter.with_weight_normalize:
            weight = normalize_weight_for_conv2d_transpose(weight, var_device=parameter.var_device)

        out = tf.nn.atrous_conv2d_transpose(
            x, weight, [x_shape[0], x_shape[1], x_shape[2], out_channels], rate,
            padding=parameter.padding)

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



