from random import randint
import numpy as np
from scipy import stats
import tensorflow as tf

from utils.debug_tools import timeit
from utils import utils
from cpcgan import CPCGAN
from data_utils import SortedNumberGenerator

def train_cpcgan(cpc_epochs, gan_epochs):
    name = 'cpcgan'
    cpcgan_args = utils.load_args()[name]
    batch_size = cpcgan_args['batch_size']
    terms = cpcgan_args['cpc']['hist_terms']
    predict_terms = cpcgan_args['cpc']['future_terms']
    image_size = cpcgan_args['image_shape'][0]
    color = cpcgan_args['color']

    train_data = SortedNumberGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, color=color, rescale=False)

    validation_data = SortedNumberGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, color=color, rescale=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    cpcgan = timeit(lambda: CPCGAN(name, cpcgan_args, sess=sess, reuse=False, log_tensorboard=True), name='CPCGAN')
    
    # cpcgan.restore_cpc()  # restore cpc only
    cpcgan.restore()      # restore the entire cpcgan
    if cpc_epochs != 0:
        train_cpc(cpcgan, cpc_epochs, train_data, validation_data)

    train_gan(cpcgan, gan_epochs, train_data, validation_data)

def train_cpc(cpcgan, epochs, train_data, validation_data):
    print('Start Training CPC.')
    i = 0
    for _ in range(epochs):
        cpcgan.log_tensorboard = True
        losses = []
        accuracies = []
        # Training
        for (history, future), label in train_data:
            cpc_run_batch(cpcgan, history, future, label,
                    'Training', i, losses, accuracies)
            i += 1
            if i % 1e3 == 0:
                break
        print('\nTraining Epoch Is Done. \nStart Validation...')

        cpcgan.log_tensorboard = False
        losses = []
        accuracies = []
        # Validation
        for j, ((history, future), label) in enumerate(validation_data):
            cpc_run_batch(cpcgan, history, future, label,
                    'Validation', j, losses, accuracies)
            if j >= 1e2:
                break
        print('\nValidation Epoch Is Done.\n Start Saving...')

def cpc_run_batch(cpcgan, history, future, label, dataset, i, losses, accuracies):
    training = True if dataset == 'Training' else False
    logits, loss = cpcgan.optimize_cpc(history, future, training, label)

    losses.append(loss)
    prob = 1 / (1 + np.exp(-logits))
    accuracy = np.sum((prob > .5) == label) / np.size(label)
    accuracies.append(accuracy)
    print('\r{} step {: 4d}:\tLoss {:2.4f}\tAccuracy {:3.2f}\t \
        Average Loss: {:2.4f}\t Average Accuracy: {:3.2f}'.format(dataset, int(i % 1e3), 
                                                                loss, accuracy, 
                                                                np.mean(losses), 
                                                                np.mean(accuracies)), end="")

def train_gan(cpcgan, epochs, train_data, validation_data):
    print('Start Training GAN. Good Luck :-)')
    i = 0
    for _ in range(epochs):
        cpcgan.log_tensorboard = True
        generator_losses = []
        critic_losses = []
        # Training
        for (history, future), label in train_data:
            gan_run_batch(cpcgan, history, future, label,
                    'Training', i, generator_losses, critic_losses)
            i += 1
            if i % 1e3 == 0:
                break
        print('\nTraining Epoch Is Done. \nStart Validation...')

        cpcgan.log_tensorboard = False
        generator_losses = []
        critic_losses = []
        # Validation
        for j, ((history, future), label) in enumerate(validation_data):
            gan_run_batch(cpcgan, history, future, label,
                    'Validation', j, generator_losses, critic_losses)
            if j >= 3e2:
                break
        print('\nValidation Epoch Is Done.\n Start Saving...')
        # self.save()
        print('Model Saved')

def gan_run_batch(cpcgan, history, future, label, dataset, i, generator_losses, critic_losses):
    training = True if dataset == 'Training' else False
    generator_loss, critic_loss = cpcgan.optimize_gan(history, future, training, label)

    generator_losses.append(generator_loss)
    critic_losses.append(critic_loss)
    print('\r{} step {:4d}:\tGenerator Loss {:2.4f}\tCritic Loss {:2.4f}\t \
        Average Generator Loss: {:2.4f}\t Average Critic Loss: {:2.4f}'.format(dataset, int(i % 1e3), 
                                                                            generator_loss, critic_loss, 
                                                                            np.mean(generator_losses), 
                                                                            np.mean(critic_losses)), end="")


if __name__ == "__main__":

    train_cpcgan(1, 100000)
