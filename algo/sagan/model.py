import os
import os.path as osp
from time import time
from collections import deque
import numpy as np
import tensorflow as tf
import tensorflow.contrib.gan as tfgan

from utility.debug_tools import timeit
from utility.utils import pwc, squarest_grid_size
from utility.image_processing import read_image, save_image, image_dataset
from basic_model.model import Model
from networks import Generator, Discriminator


class SAGAN(Model):
    """ Interface """
    def __init__(self, 
                 name, 
                 args,
                 training=True,
                 sess_config=None, 
                 save=False, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        self._training = training
        self.batch_size = args['batch_size']
        self.image_shape = args['image_shape']
        self.train_dir = args['train_dir']
        self.results_dir = args['results_dir']

        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

    def evaluate(self, n_iterations=1):
        for i in range(n_iterations):
            t, image = timeit(lambda: self.sess.run(self.gen_image))
            print(f'\rEvaluation Time: {t}')
            save_image(image, f'{self.results_dir}/eval_{i}.png')

    def train(self):
        start = time()
        times = deque(maxlen=100)

        for i in range(1, self.args['n_iterations'] + 1):
            if self._time_to_save(i, interval=500):
                t1, (_, summary) = timeit(lambda: self.sess.run([self.dis_opt_op, self.graph_summary]))
                t2, (_, gen_image) = timeit(lambda: self.sess.run([self.gen_opt_op, self.gen_image]))
                t = t1 + t2
                times.append(t)
                print(f'\rTraining Time: {(time() - start) / 60:.2f}m; Iterator {i};\t\
                        Average {np.mean(times):.3F} seconds per pass',
                    end='')
                print()
                self.writer.add_summary(summary, i)
                save_image(gen_image, f'{self.results_dir}/{i:0>6d}.png')
                self.save()
            else:
                t1, _ = timeit(lambda: self.sess.run([self.dis_opt_op]))
                t2, _ = timeit(lambda: self.sess.run(self.gen_opt_op))
                t = t1 + t2
                times.append(t)
                print(f'\rTraining Time: {(time() - start) / 60:.2f}m; Iterator {i};\t\
                        Average {np.mean(times):.3F} seconds per pass',
                    end='')

    def _build_graph(self):
        with tf.device('/CPU: 0'):
            self.image = self._prepare_data()
        gen_args = self.args['generator']
        gen_args['batch_size'] = self.batch_size
        self.generator = Generator('Generator', 
                                    gen_args, 
                                    self.graph, 
                                    self.training,
                                    scope_prefix= self.name, 
                                    log_tensorboard=self.log_tensorboard,
                                    log_params=self.log_params)
        self.gen_image = self.generator.image
        dis_args = self.args['discriminator']
        self.real_discriminator = Discriminator('Discriminator', 
                                                dis_args, 
                                                self.graph, 
                                                self.image,
                                                False,
                                                self.training,
                                                scope_prefix= self.name,
                                                log_tensorboard=self.log_tensorboard,
                                                log_params=self.log_params)
        self.fake_discriminator = Discriminator('Discriminator',
                                                dis_args,
                                                self.graph,
                                                self.gen_image,
                                                False,
                                                self.training,
                                                scope_prefix=self.name,
                                                log_tensorboard=False,
                                                log_params=False,
                                                reuse=True)
        
        self.gen_loss = self._generator_loss()
        self.dis_loss = self._discriminator_loss()

        self.gen_opt_op, _, _ = self.generator._optimization_op(self.gen_loss)
        self.dis_opt_op, _, _ = self.real_discriminator._optimization_op(self.dis_loss)
        
        with tf.device('/CPU: 0'):
            self._log_train_info()

    def _prepare_data(self):
        with tf.name_scope('image'):
            _, image = image_dataset(self.train_dir, self.batch_size, self.image_shape[:-1], norm_range=[-1, 1])
        return image

    def _generator_loss(self):
        with tf.name_scope('gen_loss'):
            loss = -tf.reduce_mean(self.fake_discriminator.logits)
        return loss

    def _discriminator_loss(self):
        with tf.name_scope('dis_loss'):
            real_loss = tf.reduce_mean(tf.nn.relu(1 - self.real_discriminator.logits))
            fake_loss = tf.reduce_mean(tf.nn.relu(1 + self.fake_discriminator.logits))
            loss = real_loss + fake_loss
        
        return loss

    def _log_train_info(self):
        num_images = min(self.batch_size, 16)
        image_shape = self.image_shape[:-1]
        image_grid = lambda vis_images: tfgan.eval.image_grid(
                           vis_images[:num_images],
                           grid_shape=squarest_grid_size(num_images),
                           image_shape=image_shape)

        def image_stats(image):
            means = tf.reduce_mean(image, 0, keep_dims=True)
            vars = tf.reduce_mean(tf.squared_difference(image, means), 0, keep_dims=True)
            mean, var = tf.reduce_mean(means), tf.reduce_mean(vars)

            return mean, var

        if self.log_tensorboard:
            with tf.name_scope('train_info'):
                tf.summary.histogram('z_', self.generator.z)
                with tf.name_scope('loss'):
                    tf.summary.scalar('generator_loss_', self.gen_loss)
                    tf.summary.scalar('discriminator_loss_', self.dis_loss)
                    tf.summary.scalar('loss_', self.gen_loss + self.dis_loss)

                with tf.name_scope('image'):
                    tf.summary.image('generated_image_', image_grid(self.gen_image), max_outputs=1)
                    tf.summary.histogram('generated_image_hist_', self.gen_image)
                    gen_mean, gen_var = image_stats(self.gen_image)
                    real_mean, real_var = image_stats(self.image)
                    tf.summary.scalar('gen_mean_', gen_mean)
                    tf.summary.scalar('gen_var_', gen_var)
                    tf.summary.scalar('real_mean_', real_mean)
                    tf.summary.scalar('real_var_', real_var)
            
                with tf.name_scope('prob'):
                    tf.summary.histogram('real_prob_hist_', self.real_discriminator.prob)
                    tf.summary.histogram('fake_prob_hist_', self.fake_discriminator.prob)
                    tf.summary.scalar('real_prob_', tf.reduce_mean(self.real_discriminator.prob))
                    tf.summary.scalar('fake_prob_', tf.reduce_mean(self.fake_discriminator.prob))
                
                with tf.name_scope('logit'):
                    tf.summary.histogram('real_logits_hist_', self.real_discriminator.logits)
                    tf.summary.histogram('fake_logits_hist_', self.fake_discriminator.logits)
                    tf.summary.histogram('real_logits_', tf.reduce_mean(self.real_discriminator.logits))
                    tf.summary.histogram('fake_logits_', tf.reduce_mean(self.fake_discriminator.logits))
