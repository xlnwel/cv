import tensorflow as tf

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
                 log_params=False):
        self.z_dim = args['z_dim']
        self.padding = args['padding']
        self.norm = args['norm']
        self.spectral_norm = args['spectral_norm']
        self.variable_scope = self._get_variable_scope(scope_prefix, name)

        super().__init__(name, 
                         args, 
                         graph, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        self.z = tf.placeholder(tf.float32, shape=(None, 1, 1, self.z_dim), name='z')
        self.image = self._net(self.z)

    def _net(self, z):
        conv = self.snconv if self.spectral_norm else self.conv
        convtrans = self.snconvtrans if self.spectral_norm else self.convtrans

        x = z
        for filters, kernel_size, strides in self.args['gen_params']:
            