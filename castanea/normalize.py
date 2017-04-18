import tensorflow as tf

def normalize_weight_for_conv2d(w, g_init, var_device='/cpu:0'):
    '''
    weight normalization for conv2d

    @param w conv2d weight to be normalized
    @param var_device device name for variable placement
    @return normalized weight
    '''
    with tf.variable_scope('weight_normalize'):
        shape = w.get_shape().as_list()
        with tf.device(var_device):
            g = tf.get_variable(name='g', shape=[1,1,1,shape[3]],
                initializer=tf.constant_initializer(g_init))

        return g * tf.nn.l2_normalize(w, [0,1,2]) 


def normalize_weight_for_conv2d_transpose(w, g_init, var_device='/cpu:0'):
    '''
    weight normalization for conv2d transpose

    @param w conv2d transpose weight to be normalized
    @param var_device device name for variable placement
    @return normalized weight
    '''
    with tf.variable_scope('weight_normalize'):
        shape = w.get_shape().as_list()
        with tf.device(var_device):
            g = tf.get_variable(name='g', shape=[1,1,shape[2],1],
                initializer=tf.constant_initializer(g_init))

        return g * tf.nn.l2_normalize(w, [0,1,3]) 

def normalize_weight_for_linear(w, g_init, var_device='/cpu:0'):
    '''
    weight normalization for linear layer

    @param w linear weight to be normalized
    @param var_device device name for variable placement
    @return normalized weight
    '''
    with tf.variable_scope('weight_normalize'):
        shape = w.get_shape().as_list()
        with tf.device(var_device):
            g = tf.get_variable(name='g', shape=[1,shape[1]],
                initializer=tf.constant_initializer(math.log(stddev)))
        return = g * tf.nn.l2_normalize(w, [0]) 

