# dataset.py
# helper functions to enable tf.data.Dataset
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

CLASS_NAMES = np.array(["Normal", "Pnemonia"])
IMG_HEIGHT = 486
IMG_WIDTH = 664

def get_onehot(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    onehot = tf.reshape(tf.one_hot(tf.where(CLASS_NAMES==parts[-2]),2), (2,))
    return onehot

# def decode_img(img, IMG_HEIGHT, IMG_WIDTH):
def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def process_path(file_path):
    label = get_onehot(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=500, BATCH_SIZE=16, AUTOTUNE=None):
    """
    This is a small dataset, only load it once, and keep it in memory.
     use `.cache(filename)` to cache preprocessing work for datasets that don't
     fit in memory.
     """
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

def prepare_dataset(ds, cache=True, shuffle_buffer_size=1000, testing_bool=True, AUTOTUNE=None):
    """
    Similar utility to prepare_for_training. Decrease batch_size to one.
    """
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
        # add optional shuffle - for testing, we do not want shuffling
        if not testing_bool:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)
            # Repeat forever
            ds = ds.repeat()
            ds = ds.batch(BATCH_SIZE)
        else:
            ds = ds.repeat() # two passes so we can scrape labels
            ds = ds.batch(1)
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds
    
def show_batch(image_batch, label_batch, BATCH_SIZE):
    """
    plot a batch of images from the Dataset iterator
    """
    plt.figure(figsize=(10,10))
    for n in range(BATCH_SIZE):
        ax = plt.subplot(5,5,n+1)
        im = image_batch[n]
        n_row, n_col, _ = im.shape
        im = np.reshape(im, (n_row, n_col))
        plt.imshow(im, cmap='gray')
        plt.title(CLASS_NAMES[tf.where(label_batch[n]==1)][0].title())
        plt.axis('off')