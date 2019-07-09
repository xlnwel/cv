import os
import os.path as osp
from time import time
from collections import deque
import numpy as np
from skimage.data import imread
from skimage.transform import resize
from skimage.io import imsave
import tensorflow as tf

from utility.debug_tools import timeit
from utility.utils import pwc
from utility.tf_utils import square_sum
from utility.image_processing import get_image, image_dataset, ImageGenerator
from basic_model.model import Model
from networks import StyleTransfer, VGG19
from utility.schedule import PiecewiseSchedule


class StyleTransferModel(Model):
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
        self.eval_image_path = args['eval_image_path']
        self.eval_image = get_image(args['eval_image_path'], image_shape=self.image_shape)
        self.style_image = get_image(args['style_image_path'], image_shape=self.image_shape)
        self.style_layers = args['style_layers']
        self.style_weights = args['style_weights']
        if not isinstance(self.style_weights, list):
            self.style_weights = [self.style_weights for _ in self.style_layers]
        self.content_weight = args['content_weight']
        self.content_layer = args['content_layer']
        self.tv_weight = args['tv_weight']

        if log_tensorboard:
            self.data_generator = ImageGenerator(self.valid_dir, self.image_shape, self.batch_size, preserve_range=False)

        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

    def evaluate(self, eval_image=False, train_time=None):
        if self.log_tensorboard:
            if train_time is None:
                raise ValueError
            summary = self.sess.run(self.graph_summary, feed_dict={self.image: self.data_generator.sample()})
            self.writer.add_summary(summary, train_time)
        if eval_image:
            t, st_image = timeit(lambda: self.sess.run(self.st_image, feed_dict={self.image: self.eval_image}))
            pwc(f'Time taking to transfer an image: {t:.2f}s', color='green')
            st_image = np.squeeze(st_image)
            # get image path
            image_filename, _ = osp.splitext(self.eval_image_path)
            _, image_filename = osp.split(image_filename)
            _, style_filename = osp.split(self.args['style_image_path'])
            results_dir = self.args['results_dir']
            if not osp.exists(results_dir):
                os.mkdir(results_dir)

            imsave(f'data/results/{image_filename}-{style_filename}', st_image, format='jpg')

    def train(self):
        start = time()
        times = deque(maxlen=100)
        for i in range(self.args['n_iterations']):
            t, _ = timeit(lambda: self.sess.run([self.opt_op]))
            times.append(t)
            if i % 100 == 0:
                self.evaluate(train_time=i)
                self.save()

            pwc(f'Iterator {i}:\t\t{(time() - start) / 60:.3f} minutes\n'
                f'Average {np.mean(times):.3F} seconds per pass', color='green')
        
    """ Implementation """
    def _build_graph(self):
        with tf.device('/CPU: 0'):
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

        self.opt_op, _, _ = self.st_net._optimization_op(self.loss)

        with tf.device('/CPU: 0'):
            self._log_loss()

    def _prepare_data(self):
        with tf.name_scope('image'):
            # Do not normalize image here, do it in StyleTransfer
            ds = image_dataset(self.train_dir, self.image_shape[:-1], self.batch_size, False)
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
        style_grams = self.sess.run(grams, feed_dict={self.st_image: self.style_image})

        losses = []
        with tf.name_scope('style_loss'):
            for i, (style_gram, gram) in enumerate(zip(style_grams, grams)):
                size = style_gram.size * self.batch_size
                losses.append(self.style_weights[i] * square_sum(style_gram - gram) / size)
            loss = tf.reduce_mean(losses)
        return loss

    def _content_loss(self, vgg, vgg_features):
        content_vgg_features = vgg(self.image, reuse=True)
        layer = self.content_layer
        size = np.prod(vgg_features[layer].shape.as_list()[1:]) * self.batch_size

        assert vgg_features[layer].shape.as_list() == content_vgg_features[layer].shape.as_list()
        with tf.name_scope('content_loss'):
            loss = self.content_weight / size * square_sum(vgg_features[layer] - content_vgg_features[layer])

        return loss

    def _tv_loss(self, st_image):
        with tf.name_scope('tv_loss'):
            tv_size_1 = np.prod(st_image[:, 1:, :, :].shape.as_list()[1:]) * self.batch_size
            tv_size_2 = np.prod(st_image[:, :, 1:, :].shape.as_list()[1:]) * self.batch_size
            return self.tv_weight * (square_sum(st_image[:, 1:, :, :] - st_image[:, :-1, :, :]) / tv_size_1
                                    + square_sum(st_image[:, :, 1:, :] - st_image[:, :, :-1, :]) / tv_size_2)

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
