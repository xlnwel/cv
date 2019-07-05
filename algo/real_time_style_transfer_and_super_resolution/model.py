from time import time
from collections import deque
import numpy as np
from skimage.data import imread
from skimage.transform import resize
import tensorflow as tf

from utility.debug_tools import timeit
from utility.utils import pwc
from utility.image_processing import image_dataset
from basic_model.model import Model
from networks import StyleTransfer, VGG19


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
        for i in range(1, self.args['n_iterations'] + 1):
            if self.log_tensorboard:
                t, (_, summary) = timeit(lambda: self.sess.run([self.opt_op, self.graph_summary]))
                times.append(t)
                if i % 100 == 0:
                    self.writer.add_summary(summary, i)
                    self.save()
            else:
                t, _ = timeit(self.sess.run([self.opt_op]))
            
            pwc(f'Iterator {i}:\t\t{(time() - start / 60):.2f} minutes\n'
                f'Average {np.mean(times)} seconds per pass', color='yellow')
        
    def _build_graph(self):
        self.image = self._prepare_data()
        self.st_net = StyleTransfer('StyleTransferNet', self.args['style_transfer'], 
                                    self.graph, self.image, scope_prefix=self.name,
                                    log_tensorboard=self.log_tensorboard,
                                    log_params=self.log_params)
        self.st_image = self.st_net.st_image

        self.vgg = VGG19(self.args['vgg_path'])
        # compute style target feature
        vgg_dict = self.vgg(self.st_image)
        grams = self._compute_gram_matrix(vgg_dict)

        self.style_loss = self._style_loss(grams)
        self.content_loss = self._content_loss(self.vgg, vgg_dict)
        self.tv_loss = self._tv_loss(self.st_image)

        with tf.name_scope('loss'):
            self.loss = self.style_loss + self.content_loss + self.tv_loss

        self.opt_op, _ = self.st_net._optimization_op(self.loss)

        self._log_loss()

    def _prepare_data(self):
        with tf.name_scope('image'):
            # Do not normalize image here, do it in StyleTransfer
            ds = image_dataset(self.train_dir, self.image_shape[:-1], self.batch_size, False)
            image = ds.make_one_shot_iterator().get_next('images')
            # image = tf.placeholder(tf.float32, [None, *self.image_shape], name='input')

        return image
        
    def _compute_gram_matrix(self, vgg_dict):
        grams = []
        with tf.name_scope('grams'):
            for layer in self.style_layers:
                _, h, w, c = vgg_dict[layer].shape
                features = tf.reshape(vgg_dict[layer], [-1, h * w, c])
                gram = tf.matmul(features, features, transpose_a=True)
                gram /= tf.cast(h * w * c, tf.float32)
                grams.append(gram)

        return grams
    
    def _style_loss(self, grams):
        with tf.Session() as sess:
            style_grams = sess.run(grams, feed_dict={self.st_net.st_image: self.style_image})

        loss = tf.constant(0.)
        with tf.name_scope('style_loss'):
            for i, (style_gram, gram) in enumerate(zip(style_grams, grams)):
                loss += self.style_weights[i] * tf.reduce_mean(tf.reduce_sum((style_gram - gram) ** 2, axis=[1, 2]), axis=0)
        
        return loss

    def _content_loss(self, vgg, vgg_dict):
        content_vgg_dict = vgg(self.image, is_reuse=True)
        layer = self.content_layer
        norm = np.prod(vgg_dict[layer].shape.as_list()[1:])
        
        with tf.name_scope('content_loss'):
            loss = self.content_weight / norm * tf.reduce_sum((vgg_dict[layer] - content_vgg_dict[layer]) ** 2)

        return loss

    def _tv_loss(self, st_image):
        with tf.name_scope('tv_loss'):
            return self.tv_weight * (tf.reduce_sum((st_image[:, 1:, :, :] - st_image[:, :-1, :, :])**2)
                                    + tf.reduce_sum((st_image[:, :, 1:, :] - st_image[:, :, :-1, :])**2))

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('style_loss_', self.style_loss)
                tf.summary.scalar('content_loss_', self.content_loss)
                tf.summary.scalar('tv_loss_', self.tv_loss)
                tf.summary.scalar('loss_', self.loss)

            with tf.name_scope('style_image'):
                tf.summary.image('st_image_', self.st_image)
