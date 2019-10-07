import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, data_dir, width, height, batch_size, loss_func='SparseCategoricalCrossentropy'):
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.data_dir = pathlib.Path(data_dir)
        self.image_count = len(list(self.data_dir.glob('*/*.jpg')))
        self.steps_per_epoch = np.ceil(self.image_count/batch_size)
        self.class_name = np.array([item.name for item in self.data_dir.glob('*')])

    def __get_label(self, path):
        parts = tf.strings.split(path, '\\')
        return parts[-2] == self.class_name

    def __labeled_for_loss(self, label):
        if self.loss_func == 'SparseCategoricalCrossentropy':
            return tf.reshape(tf.where(label), [1])[0]
        elif self.loss_func == 'CategoricalCrossentropy':
            return label
        else:
            raise ValueError('loss_func required the value of "SparseCategoricalCrossentropy" \
                        or "CategoricalCrossentropy", but got  %s.' % self.loss_func)

    def __decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img / 255.
        return tf.image.resize(img, [self.width, self.height])

    def __process_path(self, path):
        label = self.__get_label(path)
        label = self.__labeled_for_loss(label)
        img = tf.io.read_file(path)
        img = self.__decode_img(img)
        return img, label

    def __prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # ds = ds.repeat(epochs)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    @property
    def data(self):
        list_ds = tf.data.Dataset.list_files(str(self.data_dir/'*/*'))
        labeled_ds = list_ds.map(self.__process_path,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return self.__prepare_for_training(labeled_ds)
