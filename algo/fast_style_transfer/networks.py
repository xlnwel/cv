import numpy as np
import scipy.io as sio
import tensorflow as tf

from utility.utils import pwc
from utility import tf_utils
from basic_model.model import Module


class StyleTransfer(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph, 
                 image, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.image = image / 255.
        self.padding = args['padding']
        self.norm = args['norm']
        self.variable_scope = f'{scope_prefix}/{name}'

        super().__init__(name, 
                         args, 
                         graph, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)
        
    def _build_graph(self):
        self.st_image = self._st_net(self.image)

    def _st_net(self, x):
        """ Style transfer net """
        if self.norm == 'batch':
            norm = tf.layers.batch_normalization
        elif self.norm == 'layer':
            norm == tf_utils.layer_norm
        elif self.norm == 'instance':
            norm = tf_utils.instance_norm
        else:
            raise NotImplementedError
        
        # ConvNet
        with tf.variable_scope('net', reuse=self.reuse):
            for i, (filters, kernel_size, strides) in enumerate(self.args['conv_params']):
                x = self.conv_norm_activation(x, filters, kernel_size, strides, padding=self.padding, norm=norm, name=f'Conv_{i}')

            # ResNet, following paper "Identity Mappings in Deep Residual Networks"
            for i in range(self.args['n_residuals']):
                x = self.conv_resnet(x, x.shape[-1], 3, 1, padding=self.padding, norm=norm, name=f'ResBlock_{i}')
            x = tf_utils.norm_activation(x, norm, tf.nn.relu)

            for i, (filters, kernel_size, strides) in enumerate(self.args['convtras_params']):
                x = self.convtrans_norm_activation(x, filters, kernel_size, strides, norm=norm, name=f'ConvTrans_{i}')

            filters, kernel_size, strides = self.args['final_conv_params']
            x = self.conv_norm_activation(x, filters, kernel_size, strides, norm=norm, activation=tf.tanh, name='FinalConv')

            x = 127.5 * x + 127.5
            
        return x

    def conv_resnet(self, x, filters, kernel_size, strides=1, padding='same', norm=None, name=None):
        y = x
        with tf.variable_scope(name):
            y = self.conv_norm_activation(y, filters, kernel_size, strides, padding=padding, norm=norm, name='ConvNormAct1')
            y = self.conv_norm_activation(y, filters, kernel_size, strides, padding=padding, norm=norm, activation=None, name='ConvNormAct2')

            x += y

        return x

# code originally from https://github.com/lengstrom/fast-style-transfer/blob/master/src/vgg.py
# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
class VGG19:
    MEAN_PIXEL = np.array([123.68, 116.779, 103.939])
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    def __init__(self, vgg_path):
        self.data = sio.loadmat(vgg_path)
        self.weights = self.data['layers'][0]


    def __call__(self, image, name='vgg', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            image = self.preprocess(image)

            net = {}
            current = image
            for i, name in enumerate(VGG19.layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = self.weights[i][0][0][0][0]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = bias.reshape(-1)
                    current = self._conv_layer(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = self._pool_layer(current)
                net[name] = current

        assert len(net) == len(VGG19.layers)
        return net

    def _conv_layer(self, x, weights, bias):
        conv = tf.nn.conv2d(x, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
        return tf.nn.bias_add(conv, bias)

    def _pool_layer(self, x):
        return tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    def preprocess(self, image):
        return image - VGG19.MEAN_PIXEL

    def unprocess(self, image):
        return image + VGG19.MEAN_PIXEL
