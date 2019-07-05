from pathlib import Path
import tensorflow as tf

    
def image_dataset(filepath, image_size, batch_size, norm=True):
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        if norm:
            image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return preprocess_image(image)

    all_image_paths = [str(f) for f in Path(filepath).glob('*/*')]
    ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(buffer_size = len(all_image_paths))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds
