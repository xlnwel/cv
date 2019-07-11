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
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False,
                 reuse=None):
        self.z_dim = args['z_dim']
        self.padding = args['padding']
        self.spectral_norm = args['spectral_norm']
        self.batch_size = args['batch_size']
        self.variable_scope = self._get_variable_scope(scope_prefix, name)

        super().__init__(name, 
                         args, 
                         graph, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params,
                         reuse=reuse)

    def _build_graph(self):
        self.z = tf.random.normal((self.batch_size, self.z_dim))
        self.image = self._net(self.z)

    def _net(self, z):
        k = 1
        def print_layer(x, layer):
            pwc(f'{self.name} {layer} Layer {k}:\t{x.shape.as_list()}', 'magenta')
        bn = tf.layers.batch_normalization
        norm_ac = lambda x: tf_utils.norm_activation(x, norm=bn, activation=tf.nn.relu)
        conv = self.snconv if self.spectral_norm else self.conv
        convtrans = self.snconvtrans if self.spectral_norm else self.convtrans
        dense = self.sndense if self.spectral_norm else self.dense

        x = dense(z, 4*4*1024)
        x = tf.reshape(x, (-1, 4, 4, 1024))
        x = norm_ac(x)
        print_layer(x, 'Dense')
        k += 1

        trans_padding = 'valid' if self.padding == 'valid' else 'same'
        for i, (filters, kernel_size, strides) in enumerate(self.args['convtrans_params']):
            with tf.variable_scope(f'Block_{i}'):
                x = convtrans(x, filters, kernel_size, strides, padding=trans_padding)
                x = norm_ac(x)
            print_layer(x, 'ConvTrans')
            k += 1
            if i in self.args['attention_layers']:
                x = self.conv_attention(x)
                print_layer(x, 'ConvAttention')
                k += 1
        
        x = conv(x, 3, 3, 1)
        x = tf.tanh(x)
        print_layer(x, 'Conv')
                    
        return x

class Discriminator(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph, 
                 image,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False,
                 reuse=None):
        self.image = image
        self.padding = args['padding']
        self.spectral_norm = args['spectral_norm']
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
        k = 1
        def print_layer(x, layer):
            pwc(f'{self.name} {layer} Layer {k}:\t{x.shape.as_list()}', 'magenta')
        bn = tf.layers.batch_normalization
        lrelu = lambda x: tf.maximum(self.args['lrelu_slope'] * x, x)
        norm_ac = lambda x: tf_utils.norm_activation(x, norm=bn, activation=lrelu)
        conv = self.snconv if self.spectral_norm else self.conv

        for i, (filters, kernel_size, strides) in enumerate(self.args['conv_params']):
            with tf.variable_scope(f'Block_{i}'):
                x = conv(x, filters, kernel_size, strides, padding=self.padding)
                x = norm_ac(x)
            print_layer(x, 'Conv')
            k += 1
            if i in self.args['attention_layers']:
                x = self.conv_attention(x)
                print_layer(x, 'ConvAttention')
                k += 1
        
        x = conv(x, 1, 4, 1, padding='valid')
        x = tf.reshape(x, (-1, 1))
        print_layer(x, 'Conv')
        prob = tf.sigmoid(x)

        return x, prob
