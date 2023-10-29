import tensorflow as tf
def vgg(input_net, training):
    kernel_init = tf.variance_scaling_initializer(scale=2.0)

    def conv_layer(net, filter_cnt, kernel_size, strides_size=1):
        net = tf.layers.conv2d(inputs=net,
                               filters=filter_cnt,
                               kernel_size=(kernel_size, kernel_size),
                               strides=(strides_size, strides_size),
                               padding='same',
                               kernel_initializer=kernel_init
                               )
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        return net
    net = input_net / 255.0 # int 형식으로 output을 내는 것을 방지한다
    net = conv_layer(net, 32, 3)
    print(net.shape)
    net = conv_layer(net, 32, 3)
    print(net.shape)
    net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(1))
    net = conv_layer(net, 64, 3)
    net = conv_layer(net, 64, 3)
    net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(2))
    net = conv_layer(net, 128, 3)
    net = conv_layer(net, 128, 3)
    net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(3))
    net = conv_layer(net, 256, 3)
    net = conv_layer(net, 256, 3)
    net = conv_layer(net, 256, 3)
    net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(4))
    net = conv_layer(net, 256, 3)
    net = conv_layer(net, 256, 3)
    net = conv_layer(net, 256, 3)
    net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(5))
    net = tf.keras.layers.GlobalAveragePooling2D()(net)
    net = tf.keras.layers.Dense(3,kernel_initializer=kernel_init)(net)

    return net

