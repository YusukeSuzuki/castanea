import tensorflow as tf
from castanea.initializers import xavier_initializer_conv2d

def highway(y, x, kernel_height, kernel_width=None, var_device='/cpu:0'):
    '''
    HighwayNet

    @param y output units from other layer with source x
    @param x original units
    @param kernel_width gate convolution kernel width
    @param kernel_height gate convolution kernel height
    @param carry_bias
    '''
    with tf.variable_scope(None, default_name='highway'):
        kernel_width = kernel_width or kernel_height
        x_shape = x.get_shape().as_list()
        y_shape = y.get_shape().as_list()

        with tf.device(var_device):
            trans_w = tf.get_variable(name='trans_weight',
                initializer=xavier_initializer_conv2d(y, kernel_height, kernel_width),
                shape=[kernel_height, kernel_width, x_shape[3], y_shape[3]])
            trans_b = tf.get_variable(name='trans_bias',
                initializer=tf.constant_initializer(0.0), shape=[y_shape[3]])

        trans = tf.nn.conv2d(x, trans_w, strides=[1,1,1,1], padding='SAME')
        trans = tf.nn.bias_add(trans, trans_b)
        trans = tf.sigmoid(trans)
        carry = tf.subtract(1.0, trans)

        return tf.add(tf.multiply(y, trans), tf.multiply(x, carry))

