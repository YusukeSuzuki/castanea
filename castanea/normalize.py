import tensorflow as tf
from castanea.utils import device_or_none

def bn_beta_gamma(target, beta_name=None, gamma_name=None, var_device=None):
    """create beta,gamma variable for batch_normalization"""
    with device_or_none(var_device):
        return (
            tf.get_variable(beta_name, shape=target.get_shape().as_list()[-1],
                initializer=tf.zeros_initializer()), 
            tf.get_variable(gamma_name, shape=target.get_shape().as_list()[-1],
                initializer=tf.ones_initializer()) )

def batch_normalization(x, parameters):
    x = tf.layers.batch_normalization(
        x, training=parameter.training, reuse=parameters.reuse)
    return x

def normalize_weight_for_conv2d(w, g_init=1.0, var_device='/cpu:0'):
    '''
    weight normalization for conv2d

    @param w conv2d weight to be normalized
    @param var_device device name for variable placement
    @return normalized weight
    '''
    with tf.variable_scope(None, default_name='weight_normalize'):
        shape = w.get_shape().as_list()
        with device_or_none(var_device):
            g = tf.get_variable(name='g', shape=[1,1,1,shape[3]],
                initializer=tf.constant_initializer(g_init))

        return g * tf.nn.l2_normalize(w, [0,1,2]) 


def normalize_weight_for_conv2d_transpose(w, g_init=1.0, var_device='/cpu:0'):
    '''
    weight normalization for conv2d transpose

    @param w conv2d transpose weight to be normalized
    @param var_device device name for variable placement
    @return normalized weight
    '''
    with tf.variable_scope(None, default_name='weight_normalize'):
        shape = w.get_shape().as_list()
        with device_or_none(var_device):
            g = tf.get_variable(name='g', shape=[1,1,shape[2],1],
                initializer=tf.constant_initializer(g_init))

        return g * tf.nn.l2_normalize(w, [0,1,3]) 

def normalize_weight_for_linear(w, g_init=1.0, var_device='/cpu:0'):
    '''
    weight normalization for linear layer

    @param w linear weight to be normalized
    @param var_device device name for variable placement
    @return normalized weight
    '''
    with tf.variable_scope(None, default_name='weight_normalize'):
        shape = w.get_shape().as_list()
        with device_or_none(var_device):
            g = tf.get_variable(name='g', shape=[1,shape[1]],
                initializer=tf.constant_initializer(g_init))
        return g * tf.nn.l2_normalize(w, [0]) 

