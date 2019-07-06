from pathlib import Path
from random import shuffle
import numpy as np
from skimage.data import imread
from skimage.transform import resize
import tensorflow as tf

from utility.debug_tools import pwc, assert_colorize

def image_dataset(filedir, image_size, batch_size, norm=True):
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        if norm:
            image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return preprocess_image(image)

    all_image_paths = [str(f) for f in Path(filedir).glob('*/*')]
    ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds = ds.shuffle(buffer_size = len(all_image_paths))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

class ImageGenerator:
    def __init__(self, filedir, image_shape, batch_size, norm=True):
        self.all_image_paths = [str(f) for f in Path(filedir).glob('*')]
        self.total_images = len(self.all_image_paths)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.norm = norm
        self.idx = 0

    def __call__(self):
        while True:
            yield self.sample()
    
    def sample(self):
        if self.idx == 0:
            shuffle(self.all_image_paths)
        
        batch_path = self.all_image_paths[self.idx: self.idx + self.batch_size]
        batch_image = [imread(path) for path in batch_path]
        batch_image = np.array([resize(img, self.image_shape) for img in batch_image], dtype=np.float32)
        self.idx += self.batch_size
        if self.idx >= self.total_images:
            self.idx = 0

        return batch_image
    