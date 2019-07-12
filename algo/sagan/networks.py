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
        self.z = tf.random.normal((self.batch_size, self.z_dim), name='z')
        self.image = self._net(self.z)

    def _net(self, z):
        bn = tf.layers.batch_normalization
        conv = self.snconv if self.spectral_norm else self.conv
        dense = self.sndense if self.spectral_norm else self.dense

        layer = lambda x: tf.reshape(dense(x, 4*4*1024), (-1, 4, 4, 1024))
        x = self.layer_norm_act(z, layer, norm=bn, name='InitialBlock')


        for i, (filters, kernel_size, strides) in enumerate(self.args['convtrans_params']):
            layer = lambda x: self.upsample_conv(x, filters, kernel_size, strides, padding=self.padding, sn=self.spectral_norm)
            x = self.layer_norm_act(x, layer, norm=bn, name=f'Block_{i}')
            if i in self.args['attention_layers']:
                x = self.conv_attention(x)
        
        x = conv(x, 3, 3, 1)
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
        bn = tf.layers.batch_normalization
        lrelu = lambda x: tf.maximum(self.args['lrelu_slope'] * x, x)
        conv = self.snconv if self.spectral_norm else self.conv

        for i, (filters, kernel_size, strides) in enumerate(self.args['conv_params']):
            layer = lambda x: conv(x, filters, kernel_size, strides, padding=self.padding)
            x = self.layer_norm_act(x, layer, norm=bn, activation=lrelu, name=f'Block_{i}')
            if i in self.args['attention_layers']:
                x = self.conv_attention(x)
        
        x = conv(x, 1, 4, 1, padding='valid')
        x = tf.reshape(x, (-1, 1))
        prob = tf.sigmoid(x)

        return x, prob
