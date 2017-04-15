import tensorflow as tf

def xavier_initializer_linear(x):
    '''
    xavier initializer for fully connected layer

    @param x input layer, shape=[minibatch_size, unit_size]
    @return initializer
    '''
    shape = x.get_shape().as_list()
    stddev=tf.sqrt(2.0/shape[1])
    return tf.truncated_normal_initializer(stddev=stddev)

def xavier_initializer_conv2d(x, kernel_height, kernel_width=None):
    '''
    xavier initializer for 2d convolutional layer

    @param x input layer, shape=[minibatch_size, height, width, channels]
    @param kernel_height height of convolution kernel
    @param kernel_width width of convolution kernel
    @return initializer
    '''
    kernel_width = kernel_width or kernel_height
    shape = x.get_shape().as_list()
    stddev=tf.sqrt(2.0/(kernel_height * kernel_width * shape[3]))
    return tf.truncated_normal_initializer(stddev=stddev)

def xavier_initializer_conv2d_transpose(x, kernel_height, kernel_width=None):
    '''
    xavier initializer for transposed 2d convolutional layer

    @param x input layer, shape=[minibatch_size, height, width, channels]
    @param kernel_height height of convolution kernel
    @param kernel_width width of convolution kernel
    @return initializer
    '''
    kernel_width = kernel_width or kernel_height
    shape = x.get_shape().as_list()
    stddev=tf.sqrt(2.0/(kernel_height * kernel_width * shape[3]))
    return tf.truncated_normal_initializer(stddev=stddev)

