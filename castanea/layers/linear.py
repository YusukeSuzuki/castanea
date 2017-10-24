from operator import mul
from functools import reduce

import tensorflow as tf
from castanea.initializers import xavier_initializer_linear
from castanea.normalize import *
from castanea.layers.parameter import BatchNormalizationParameter, LayerParameter
from castanea.utils import device_or_none

def linear(x, shape, parameter=None):
    parameter = parameter or LayerParameter()

    with tf.variable_scope(None, default_name='linear'):
        x_shape = x.get_shape().as_list()

        x_units = reduce(mul, x_shape[1:], 1)
        out_units = reduce(mul, shape[1:], 1)

        x = tf.reshape(x, [x_shape[0], x_units])

        with device_or_none(parameter.var_device):
            weight = tf.get_variable(
                shape=[x_units, out_units], initializer=xavier_initializer_linear(x),
                name='weight')

        if parameter.with_weight_normalize:
            weight = normalize_weight_for_linear(weight, var_device=parameter.var_device)

        out = tf.matmul(x, weight)

        if parameter.with_bias:
            with device_or_none(parameter.var_device):
                bias = tf.get_variable(
                    shape=[out_units], initializer=tf.zeros_initializer(), name='bias')

            out = tf.nn.bias_add(out, bias)

        if parameter.with_batch_normalize:
            reuse = False
            if type(parameter.with_batch_normalize) is BatchNormalizationParameter:
                reuse = parameter.with_weight_normalize. reuse
            out = tf.layers.batch_normalization(out, training=parameter.training, reuse=reuse)

        if parameter.rectifier:
            out = parameter.rectifier(out)

        out = tf.reshape(out, shape)

        return out

