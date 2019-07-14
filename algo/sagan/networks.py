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
        self.z = tf.random.uniform((self.batch_size, self.z_dim), minval=-1, maxval=1, name='z')
        self.image = self._net(self.z)

    def _net(self, z):
        bn = lambda x: tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=self.training)
        relu = tf.nn.relu
        dense = self.sndense if self.spectral_norm else self.dense
        conv = self.snconv if self.spectral_norm else self.conv

        def resblock(x, filters, i):
            with tf.variable_scope(f'Block_{i}'):
                y = x
                y = relu(bn(y))
                y = self.upsample_conv(y, filters, 3, padding=self.padding, 
                                       sn=self.spectral_norm, use_bias=False,
                                       name='ResUpConv')
                y = relu(bn(y))
                y = conv(y, filters, 3, 1, padding=self.padding, name='Conv')

                x = self.upsample_conv(x, filters, 1, padding='VALID', 
                                       sn=self.spectral_norm, use_bias=False,
                                       name='UpConv1x1')
                return x + y

        # Layer definition starts here
        x = dense(z, 4*4*1024, name='InitialLayer')
        x = tf.reshape(x, [-1, 4, 4, 1024])

        x = resblock(x, 1024, 1)        # [None, 8, 8, 1024]
        x = resblock(x, 512, 2)         # [None, 16, 16, 512]
        x = resblock(x, 256, 3)         # [None, 32, 32, 256]
        x = self.conv_attention(x, downsample=True)
        x = resblock(x, 128, 4)         # [None, 64, 64, 128]
        x = resblock(x, 64, 5)          # [None, 128, 128, 64]
        x = relu(bn(x))

        x = conv(x, 3, 3, 1, padding=self.padding, name='FinalLayer')
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
        downsample = lambda x, name: tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=name)

        def initblock(x, filters):
            with tf.variable_scope('InitialBlock'):
                y = x
                y = conv(y, filters, 3, 1, padding=self.padding, name='ResConv1')
                y = lrelu(y)
                y = conv(y, filters, 3, 1, padding=self.padding, name='ResConv2')
                y = downsample(y, name='ResDownsample')

                x = downsample(x, name='Downsample')
                x = conv(x, filters, 1, 1, padding='valid', name='Conv1x1')
            
            return x + y

        def resblock(x, filters, todownsample, i):
            C = x.shape.as_list()[-1]
            # Notice that no batch normalization is used
            with tf.variable_scope(f'Block_{i}'):
                y = x
                y = lrelu(y)
                y = conv(y, filters, 3, padding=self.padding, name='ResConv1')
                y = lrelu(bn(y))
                y = conv(y, filters, 3, 1, padding=self.padding, name='ResConv2')
                if todownsample:
                    y = downsample(y, 'ResDownsample')

                if todownsample or C != filters:
                    x = conv(x, filters, 1, padding='VALID', name='Conv')
                    if downsample:
                        x = downsample(x, 'Downsample')

                return x + y

        # Layer definition starts here
        # x = [None, 128, 128, 3]
        x = initblock(x, 64)                                # [None, 64, 64, 64]
        x = resblock(x, 128, True, 1)                       # [None, 32, 32, 128]
        x = self.conv_attention(x, downsample=True)
        x = resblock(x, 256, True, 2)                       # [None, 16, 16, 256]
        x = resblock(x, 512, True, 3)                       # [None, 8, 8, 512]
        x = resblock(x, 1024, True, 4)                      # [None, 4, 4, 1024]
        x = resblock(x, 1024, False, 5)                     # [None, 4, 4, 1024]
        x = lrelu(x)
        x = tf.reduce_sum(x, [1, 2])
        out = dense(x, 1, name='FinalLayer')
        if hasattr(self, 'label'):
            assert_colorize(hasattr(self, 'n_classes'))
            y = self.embedding(self.label, self.n_classes, 1024, True)
            y = tf.reduce_mean(x * y, axis=1, keep_dims=True)
            out += y

        prob = tf.sigmoid(out)

        return out, prob
