from time import time
from collections import deque
import numpy as np
from skimage.data import imread
from skimage.transform import resize
import tensorflow as tf

from utility.debug_tools import timeit
from utility.utils import pwc
from utility.tf_utils import stats_summary
from utility.image_processing import image_dataset
from basic_model.model import Model
from networks import StyleTransfer, VGG19
from utility.image_processing import ImageGenerator


class RTSTSRModel(Model):
    """ Interface """
    def __init__(self, 
                 name, 
                 args,
                 sess_config=None, 
                 save=False, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        self.batch_size = args['batch_size']
        self.image_shape = args['image_shape']
        self.train_dir = args['train_dir']
        self.valid_dir = args['valid_dir']
        style_image = imread(args['style_image_path'])
        self.style_image = np.expand_dims(resize(style_image, self.image_shape), 0)
        self.style_layers = args['style_layers']
        self.style_weights = args['style_weights']
        if not isinstance(self.style_weights, list):
            self.style_weights = [self.style_weights for _ in self.style_layers]
        self.content_weight = args['content_weight']
        self.content_layer = args['content_layer']
        self.tv_weight = args['tv_weight']

        self.data_generator = ImageGenerator(self.train_dir, self.image_shape, self.batch_size, norm=False)
        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

    def train(self):
        start = time()
        times = deque(maxlen=100)
        for i in range(self.args['n_iterations']):
            if self.log_tensorboard:
                t, (_, summary) = timeit(lambda: self.sess.run([self.opt_op, self.graph_summary]))
                times.append(t)
                if i % 100 == 0:
                    self.writer.add_summary(summary, i)
                    self.save()
            else:
                t, _ = timeit(self.sess.run([self.opt_op]))
            
            pwc(f'Iterator {i}:\t\t{(time() - start) / 60:.3f} minutes\n'
                f'Average {np.mean(times):.3F} seconds per pass', color='yellow')
        
    def _build_graph(self):
        self.image = self._prepare_data()
        self.st_net = StyleTransfer('StyleTransferNet', self.args['style_transfer'], 
                                    self.graph, self.image, scope_prefix=self.name,
                                    log_tensorboard=self.log_tensorboard,
                                    log_params=self.log_params)
        self.st_image = self.st_net.st_image

        self.vgg = VGG19(self.args['vgg_path'])
        
        vgg_features = self.vgg(self.st_image)
        grams = self._compute_gram_matrix(vgg_features)

        self.style_loss = self._style_loss(grams)
        self.content_loss = self._content_loss(self.vgg, vgg_features)
        self.tv_loss = self._tv_loss(self.st_image)

        with tf.name_scope('loss'):
            self.loss = self.style_loss + self.content_loss + self.tv_loss

        self.opt_op, _ = self.st_net._optimization_op(self.loss)

        self._log_loss()

    def _prepare_data(self):
        with tf.name_scope('image'):
            # Do not normalize image here, do it in StyleTransfer
            # ds = image_dataset(self.train_dir, self.image_shape[:-1], self.batch_size, False)
            sample_type = (tf.float32)
            sample_shape = (None, *self.image_shape)
            ds = tf.data.Dataset.from_generator(self.data_generator, sample_type, sample_shape)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            image = ds.make_one_shot_iterator().get_next('images')

        return image
        
    def _compute_gram_matrix(self, vgg_features):
        grams = []
        with tf.name_scope('grams'):
            for layer in self.style_layers:
                _, h, w, c = vgg_features[layer].shape
                features = tf.reshape(vgg_features[layer], [-1, h * w, c])
                gram = tf.matmul(features, features, transpose_a=True)
                gram /= tf.cast(h * w * c, tf.float32)
                grams.append(gram)

        return grams
    
    def _style_loss(self, grams):
        with tf.Session() as sess:
            style_grams = sess.run(grams, feed_dict={self.st_net.st_image: self.style_image})

        losses = []
        with tf.name_scope('style_loss'):
            for i, (style_gram, gram) in enumerate(zip(style_grams, grams)):
                losses.append(self.style_weights[i] * 2 * tf.nn.l2_loss(style_gram - gram) / style_gram.size)
            loss = tf.reduce_mean(losses)
        return loss

    def _content_loss(self, vgg, vgg_features):
        content_vgg_features = vgg(self.image, reuse=True)
        layer = self.content_layer
        size = np.prod(vgg_features[layer].shape.as_list()[1:])

        assert vgg_features[layer].shape.as_list() == content_vgg_features[layer].shape.as_list()
        with tf.name_scope('content_loss'):
            loss = self.content_weight / size * 2 * tf.nn.l2_loss(vgg_features[layer] - content_vgg_features[layer])

        return loss

    def _tv_loss(self, st_image):
        with tf.name_scope('tv_loss'):
            tv_size_1 = np.prod(st_image[:, 1:, :, :].shape.as_list()[1:])
            tv_size_2 = np.prod(st_image[:, :, 1:, :].shape.as_list()[1:])
            return self.tv_weight * 2 * (tf.nn.l2_loss(st_image[:, 1:, :, :] - st_image[:, :-1, :, :]) / tv_size_1
                                        + tf.nn.l2_loss(st_image[:, :, 1:, :] - st_image[:, :, :-1, :]) / tv_size_2)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('style_loss_', self.style_loss)
                tf.summary.scalar('content_loss_', self.content_loss)
                tf.summary.scalar('tv_loss_', self.tv_loss)
                tf.summary.scalar('loss_', self.loss)

            with tf.name_scope('style_image'):
                tf.summary.image('original_image_', self.image, max_outputs=1)
                tf.summary.image('st_image_', self.st_image, max_outputs=1)
                tf.summary.histogram('st_image_hist_', self.st_image)
                style = tf.constant(self.style_image)
                tf.summary.image('style_image_', style)
                tf.summary.histogram('style_image_hist_', self.style_image)
