import tensorflow as tf

from utility.debug_tools import assert_colorize, pwc
from utility.utils import pwc
from utility import tf_utils
from basic_model.model import Module


class Generator(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph, 
                 training,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False,
                 reuse=None):
        self.z_dim = args['z_dim']
        self.padding = args['padding']
        self.spectral_norm = args['spectral_norm']
        self.batch_size = args['batch_size']

        self._training = training   # argument 'training' for batch normalization
        self.variable_scope = self._get_variable_scope(scope_prefix, name)

        super().__init__(name, 
                         args, 
                         graph, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params,
                         reuse=reuse)

    def _build_graph(self):
        self.z = tf.random.uniform((self.batch_size, 1, 1, self.z_dim), minval=-1, maxval=1, name='z')
        self.image = self._net(self.z)

    def _net(self, z):
        bn = lambda x: tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=self.training)
        relu = tf.nn.relu
        dense = self.sndense if self.spectral_norm else self.dense
        convtrans = self.snconvtrans if self.spectral_norm else self.convtrans
        def block(x, filters, i):
            with tf.variable_scope(f'Block_{i}'):
                x = self.upsample_conv(x, filters, 3, padding=self.padding, 
                                       sn=self.spectral_norm, use_bias=False, 
                                       name='UpsampleConv')
                x = relu(bn(x))
            return x

        # Layer definition starts here
        # z: [None, z_dim]
        x = convtrans(z, 1024, 4, 1, use_bias=False)
        x = relu(bn(x))

        x = block(x, 512, 1)            # [None, 8, 8, 512]
        x = block(x, 256, 2)            # [None, 16, 16, 256]
        x = self.conv_attention(x, downsample=False)
        x = block(x, 128, 3)            # [None, 32, 32, 128]
        x = block(x, 64, 4)             # [None, 64, 64, 64]

        x = self.upsample_conv(x, 3, 3, 1, padding=self.padding, sn=self.spectral_norm, name='FinalLayer')
        x = tf.tanh(x)

        return x

class Discriminator(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph, 
                 image,
                 training,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False,
                 reuse=None):
        self.image = image
        self.padding = args['padding']
        self.spectral_norm = args['spectral_norm']

        self._training = training   # argument 'training' for batch normalization
        self.variable_scope = self._get_variable_scope(scope_prefix, name)

        super().__init__(name, 
                         args, 
                         graph, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params,
                         reuse=reuse)

    def _build_graph(self):
        self.logits, self.prob = self._net(self.image)

    def _net(self, x):
        bn = lambda x: tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=self.training)
        lrelu = lambda x: tf.nn.leaky_relu(x, self.args['lrelu_slope'])
        conv = self.snconv if self.spectral_norm else self.conv
        dense = self.sndense if self.spectral_norm else self.dense
        def block(x, filters, i):
            with tf.variable_scope(f'Block_{i}'):
                x = conv(x, filters, 4, 2, padding=self.padding, 
                         use_bias=False, name='DownsampleConv')
                x = lrelu(bn(x))
            return x

        # Layer definition starts here
        # x = [None, 128, 128, 3]
        x = conv(x, 64, 4, 2, padding=self.padding)         # [None, 64, 64, 64]
        x = lrelu(x)
        x = block(x, 128, 1)                                # [None, 32, 32, 128]
        x = block(x, 256, 2)                                # [None, 16, 16, 256]
        x = self.conv_attention(x, downsample=False)
        x = block(x, 512, 3)                                # [None, 8, 8, 512]
        x = block(x, 1024, 5)                               # [None, 4, 4, 1024]
        x = conv(x, 4, 1, padding='valid', use_bias=False)
        prob = tf.sigmoid(tf.reduce_mean(x, [1, 2, 3]))

        return x, prob
