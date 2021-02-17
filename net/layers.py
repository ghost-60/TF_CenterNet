import tensorflow as tf

def _bn(inputs, is_training):
    bn = tf.layers.batch_normalization(
        inputs=inputs,
        training=is_training,
        momentum = 0.99
    )
    return bn

def _conv_nn(inputs, filters, kernel_size, strides=1, padding='same', activation=None, is_training=False, use_bn=True, name = "w99"):
    filter_ = tf.get_variable(name, shape=[kernel_size[0], kernel_size[1], inputs.shape[-1], filters], initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(inputs, filter_, [1, strides, strides, 1], padding=padding)
    if activation is not None:
        conv = activation(conv)
    return conv

def _conv(inputs, filters, kernel_size, strides=1, padding='same', activation=tf.nn.relu, is_training=False, use_bn=True):
    if use_bn:
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias = False
        )
        conv = _bn(conv, is_training)
    else:
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )
    if activation is not None:
        conv = activation(conv)
    return conv

def upsampling(inputs, method):
    #assert method in ["resize", "deconv"]

    #if method == "resize":
    #input_shape = tf.shape(inputs)
    #output = tf.image.resize_nearest_neighbor(inputs, (input_shape[1] * 2, input_shape[2] * 2))

    # if method == "deconv":
    numm_filter = inputs.shape.as_list()[-1]
    output = tf.layers.Conv2DTranspose(
        #inputs=inputs,
        filters=numm_filter,
        kernel_size=4,
        strides=2,
        padding='SAME'
    ) (inputs)
    return output

