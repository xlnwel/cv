import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utility import tf_utils
from basic_model.model import Model


class CPC(Model):
    """ Interface """
    def __init__(self, name, args, batch_size, image_shape, 
                 code_size, x_future=None, training=False,
                 sess=None, reuse=False, build_graph=True, 
                 log_tensorboard=False, loss_type='supervised', 
                 scope_prefix=''):
        self.code_size = code_size
        self.context_size = args['context_size']
        self.hist_terms = args['hist_terms']
        self.future_terms = args['future_terms']
        self.image_shape = image_shape
        self.batch_size = batch_size
        self._loss_type = loss_type
        self.x_future = x_future
 
        self._training = training
        self._variable_scope = f'{scope_prefix}/{name}'

        super().__init__(name, args, sess=sess, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)
        
    @property
    def global_variables(self):
        return tf.global_variables(scope=self._variable_scope)
        
    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self._variable_scope)

    """ Implementation """
    def _build_graph(self):
        self.x_history, self.x_future, self.label, self.x = self._setup_placeholder()

        x_history_flat = tf.reshape(self.x_history, [-1, *self.image_shape], name='x_history_flat')
        x_future_flat = tf.reshape(self.x_future, [-1, *self.image_shape], name='x_future_flat')
        z_history_flat = self._encode(x_history_flat)
        z_future_flat = self._encode(x_future_flat, reuse=True)

        z_history = tf.reshape(z_history_flat, [-1, self.hist_terms, self.code_size], name='z_history')
        z_future = tf.reshape(z_future_flat, [-1, self.future_terms, self.code_size], name='z_future')
        
        self.context = self._autoregressive(z_history)

        if self._loss_type == 'supervised':
            self.predictions, self.individual_logits, self.logits, self.loss = self._loss(self.context, z_future)
        else:
            self.loss = self._loss(self.context, z_future)

    def _setup_placeholder(self):
        with tf.name_scope('placeholder'):
            x_history = tf.placeholder(tf.float32, [None, self.hist_terms, *self.image_shape], name='x_history')
            x_future = (tf.identity(self.x_future, name='x_future') if self.x_future is not None else
                            tf.placeholder(tf.float32, [None, self.future_terms, *self.image_shape], name='x_future'))
            label = tf.placeholder(tf.int32, [None, 1], name='label')
            x = tf.placeholder(tf.float32, [None, *self.image_shape], 'x')

            return x_history, x_future, label, x

    def _encode(self, x, reuse=None):
        with tf.variable_scope('encoder', reuse=self._reuse if reuse is None else reuse):
            bn = tf.layers.batch_normalization
            leaky_relu = lambda x: tf.nn.leaky_relu(x, 0.3)

            x = self._conv_norm_activation(x, 64, 3, 2, padding='valid', normalization=bn, activation=leaky_relu)
            x = self._conv_norm_activation(x, 64, 3, 2, padding='valid', normalization=bn, activation=leaky_relu)
            x = self._conv_norm_activation(x, 64, 3, 2, padding='valid', normalization=bn, activation=leaky_relu)

            x = tf.reshape(x, [-1, 2 * 2 * 64])
            x = self._dense_norm_activation(x, 256, normalization=bn, activation=leaky_relu)
            x = self._dense(x, self.code_size)

        return x

    def _autoregressive(self, x):
        with tf.variable_scope('autoregressive', reuse=self._reuse):
            cell = tc.rnn.GRUCell(self.context_size, name='ar_context')

            initial_state = cell.zero_state(self.batch_size, tf.float32)

            outputs, _ = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

        return outputs[:, -1]

    def _loss(self, context, z_future):
        if self._loss_type == 'supervised':
            return self._supervised(context, z_future)
        elif self._loss_type == 'dim':
            return self._dim(context, z_future)

    def _supervised(self, context, z_future):
        predictions = []
        with tf.variable_scope('prediction', reuse=self._reuse):
            # simple dense, one for each future term
            for _ in range(self.future_terms):
                x = self._dense(context, self.code_size)
                predictions.append(x)
            predictions = tf.stack(predictions, axis=1)

        with tf.name_scope('loss'):
            individual_logits = tf.reduce_mean(predictions * z_future, axis=-1)
            logits = tf.reduce_mean(individual_logits, axis=-1, keepdims=True)

            loss = tf.losses.sigmoid_cross_entropy(self.label, logits)

        if self.log_tensorboard:
            tf.summary.scalar('loss_', loss)

        return predictions, individual_logits, logits, loss

    """ I've modified some code above, following code requires some further modifications to make it work again """
    def _dim(self, context, z):
        with tf.variable_scope('loss'):
            losses = []
            for i in range(self.future_terms):
                E_joint, E_prod = self._score(context, z[:, i, :])

                MI = E_joint - E_prod

                if self.log_tensorboard:
                    tf.summary.scalar('Local_MI_{}'.format(i), MI)

                losses.append(-MI)
            loss = tf.reduce_mean(losses)

            if self.log_tensorboard:
                tf.summary.scalar('loss_', loss)

        return loss

    def _score(self, context, z):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            T_joint = self._get_score(context, z)
            T_prod = self._get_score(context, z, shuffle=True)

            log2 = np.log(2.)
            E_joint = tf.reduce_mean(log2 - tf.math.softplus(tf.negative(T_joint)))
            E_prod = tf.reduce_mean(tf.math.softplus(tf.negative(T_prod)) + T_prod - log2)

        return E_joint, E_prod

    def _get_score(self, context, z, shuffle=False):
        with tf.variable_scope('score', reuse=tf.AUTO_REUSE):
            if shuffle:
                context = self._local_shuffle(context)
                context = tf.stop_gradient(context)
            
            context = tf.concat([context, z], axis=-1)
            
            scores = self._local_discriminator(context)

        return scores

    def _local_discriminator(self, x):
        with tf.variable_scope('discriminator_net', reuse=tf.AUTO_REUSE):
            x = self._dense_norm_activation(x, 512, activation=tf.nn.relu)
            x = self._dense_norm_activation(x, 256, activation=tf.nn.relu)
            x = self._dense(x, 1)

        return x

    def _local_shuffle(self, x):
        with tf.name_scope('local_shuffle'):
            _, d1 = x.shape
            d0 = self.batch_size
            b = tf.random_uniform(tf.stack([d0, d1]))
            idx = tc.framework.argsort(b, 0)
            idx = tf.reshape(idx, [-1])
            adx = tf.range(d1)
            adx = tf.tile(adx, [d0])

            x = tf.reshape(tf.gather_nd(x, tf.stack([idx, adx], axis=1)), (d0, d1))

        return x
