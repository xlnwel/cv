import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utility.utils import scope_name
from utility import tf_utils
from basic_model.model import Module

class WGANGP(Module):
    """ Interface """
    def __init__(self, name, args, batch_size, image_shape, 
                 code_size, code, image, training=False,
                 reuse=False, build_graph=True, 
                 log_tensorboard=False, scope_prefix=''):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.code_size = code_size
        self.noise_size = args['noise_size']
        self.z_size = self.code_size + self.noise_size
        self.code = code
        self.image = image

        self._training = training
        self.critic_coeff = args['critic_coeff']

        self._variable_scope = f'{scope_prefix}/{name}'
                
        super().__init__(name, args, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)

    @property
    def global_variables(self):
        return tf.global_variables(scope=self._variable_scope)

    @property
    def trainable_variables(self):
        return self.generator_trainable_variables + self.critic_trainable_variables

    @property
    def generator_trainable_variables(self):
        return self.generator.trainable_variables

    @property
    def critic_trainable_variables(self):
        return self.real_critic.trainable_variables

    """ Implementation """
    def _build_graph(self):
        # if you'd like to customize the noise vector, uncomment the following line and,,, good luck :-)
        # self.noise = tf.placeholder(tf.float32, [None, self.noise_size], name='noise')
        self.noise = tf.truncated_normal([self.batch_size, self.noise_size], name='noise')
        

        self.z = tf.concat([self.code, self.noise], axis=1, name='z')

        self.generator = Generator('generator', self._generator_args(), reuse=self._reuse, scope_prefix=self._variable_scope)
        
        # interpolated image
        t = np.random.random(size=(self.batch_size, 1, 1, 1))
        with tf.name_scope('interpolated_image'):
            self.interpolated_image = t * self.generator.generated_image + (1 - t) * self.image

        self.real_critic = Critic('critic', self._critic_args(self.image), reuse=self._reuse, scope_prefix=self._variable_scope)
        self.fake_critic = Critic('critic', self._critic_args(self.generator.generated_image), reuse=True, scope_prefix=self._variable_scope)
        self.interpolated_critic = Critic('critic', self._critic_args(self.interpolated_image), reuse=True, scope_prefix=self._variable_scope)

        generator_loss, wasserstein_loss = self._wasserstein_loss(self.real_critic.validity, self.fake_critic.validity)
        self.generator_loss = generator_loss
        self.critic_loss = (wasserstein_loss + self.critic_coeff 
                            * self._gradient_penalty(self.interpolated_critic.validity, self.interpolated_image))

    def _generator_args(self):
        args = {
            'image_shape': self.image_shape,
            'z_size': self.z_size,
            'z': self.z,
            'training': self._training,
        }

        return args

    def _critic_args(self, image):
        args = {
            'image_shape': self.image_shape,
            'image': image,
            'training': self._training,
        }

        return args

    def _wasserstein_loss(self, real, fake):
        # written according to https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
        real_mean = tf.reduce_mean(real)
        generator_loss = -tf.reduce_mean(fake, name='generator_loss')
        wasserstein_loss = tf.negative(real_mean + generator_loss, 'wasserstein_loss')
        
        return generator_loss, wasserstein_loss

    def _gradient_penalty(self, validity, interpolated_image):
        # written according to https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
        interpolated_grads = tf.gradients(validity, interpolated_image, name='interpolated_grads')

        grads_l2 = tf.sqrt(tf.reduce_sum(tf.square(interpolated_grads)), name='grads_l2')

        return tf.square(grads_l2 - 1, name='gradient_penalty')


class Generator(Module):
    """ Interface """
    def __init__(self, name, args, reuse=False, scope_prefix=''):
        self.image_shape = args['image_shape']
        self.z_size = args['z_size']
        self.z = args['z']
        self._training = args['training']
        self._variable_scope = scope_name(scope_prefix, name)

        super().__init__(name, args, reuse)

    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self._variable_scope)

    """ Implementation """
    def _build_graph(self):
        self.generated_image = self._build_generator(self.z)
    
    def _build_generator(self, x):
        bn = lambda x: tf.layers.batch_normalization(x, momentum=.8)
        relu = lambda x: tf.nn.relu(x)

        x = self._dense_norm_activation(x, 7 * 7 * 128, activation=relu)
        x = tf.reshape(x, [-1, 7, 7, 128])
        x = tf.image.resize_images(x, (14, 14))
        x = self._conv_norm_activation(x, 128, 4)
        x = relu(bn(x))
        x = tf.image.resize_images(x, (28, 28))
        x = self._conv_norm_activation(x, 64, 4)
        x = relu(bn(x))
        x = self._conv_norm_activation(x, self.image_shape[-1], 4, activation=tf.tanh)

        return x


class Critic(Module):
    """ Interface """
    def __init__(self, name, args, reuse=False, scope_prefix=''):
        self.image_shape = args['image_shape']
        self.image = args['image']
        self._training = args['training']
        self._variable_scope = scope_name(scope_prefix, name)

        super().__init__(name, args, reuse)

    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self._variable_scope)

    """ Implementation """
    def _build_graph(self):
        self.validity = self._build_critic(self.image)

    def _build_critic(self, x):
        bn = lambda x: tf.layers.batch_normalization(x, momentum=.8)
        leaky_relu = lambda x: tf.nn.leaky_relu(x, 0.2)
        dropout = lambda x: tf.layers.dropout(x, .25)

        x = self._conv_norm_activation(x, 16, 3, 2, activation=leaky_relu)
        x = dropout(x)
        x = self._conv_norm_activation(x, 32, 3, 2)
        x = tf.image.pad_to_bounding_box(x, 0, 0, 8, 8)
        x = leaky_relu(bn(x))
        x = dropout(x)
        x = self._conv_norm_activation(x, 64, 3, 2)
        x = leaky_relu(bn(x))
        x = dropout(x)
        x = self._conv_norm_activation(x, 128, 3, 1)
        x = leaky_relu(bn(x))
        x = dropout(x)
        x = tf.reshape(x, [-1, 4 * 4 * 128])
        x = self._dense_norm_activation(x, 1)

        return x
