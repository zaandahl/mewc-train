import os, os.path, yaml, imgaug
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imgaug import augmenters as iaa

imgaug.seed(12345) # set the global aug seed

# Reads a yaml file and returns a dictionary
def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# Prints the number of images in the dataset and the number of images in each class
def print_dsinfo(ds_df, ds_name='default'):
    print('Dataset: ' + ds_name)
    print(f'Number of images in the dataset: {ds_df.shape[0]}')
    print(ds_df['Label'].value_counts())
    print(f'\n')
    
# Used to create a sampled train or validation/test sets when the latter comes from a different pool of images
def create_train(ds_path, seed=42, ns=1000):
    
    # Creates a sampled pandas dataframe with the image path and class label derived from the directory structure
    def create_dataframe(ds_path, n, seed):
        dir_ = Path(ds_path) # Selecting folder paths in dataset
        ds_filepaths = list(dir_.glob(r'**/*.jpg'))
        ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths)) # Mapping labels...
        ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str) # Data set paths & labels
        ds_labels = pd.Series(ds_labels, name='Label')
        ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
        
        ds_df = ds_df.sample(frac=1, random_state=seed).reset_index(drop=True) # Randomising and resetting indexes
        ds_df = ds_df.groupby('Label').apply(lambda x: x.sample(n=n, replace=len(x) < n)) # Sampling n images from each class
        return ds_df
    
    train_df = create_dataframe(ds_path, ns, seed)
    num_classes = train_df['Label'].nunique()
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True) # Randomise and reset indexes
    return(train_df, num_classes)

# Creates a fixed dataset (uses all images in the target directory)
def create_fixed(ds_path):
    # Selecting folder paths in dataset
    dir_ = Path(ds_path)
    ds_filepaths = list(dir_.glob(r'**/*.jpg'))
    # Mapping labels...
    ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths))
    # Data set paths & labels
    ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str)
    ds_labels = pd.Series(ds_labels, name='Label')
    # Concatenating...
    ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
    return ds_df

# This function takes a pandas df from create_dataframe and converts to a TensorFlow dataset
def create_tensorset(in_df, img_size, batch_size, magnitude, ds_name="train"):
    
    # helper function to use with the lambda mapping
    def load(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        img = tf.image.resize(img, size=(img_size, img_size))
        return img
    
    def augment(images):
        # Input to `augment()` is a TensorFlow tensor which
        # is not supported by `imgaug`. This is why we first
        # convert it to its `numpy` variant.
        images = tf.cast(images, tf.uint8)
        return rand_aug(images=images.numpy())

    in_path = in_df['File']
    in_class = LabelEncoder().fit_transform(in_df['Label'].values)

    in_class = in_class.reshape(len(in_class), 1)
    in_class = OneHotEncoder(sparse=False).fit_transform(in_class)
    rand_aug = iaa.RandAugment(n=3, m=magnitude)

    # convert to dataset
    if ds_name == "train" or ds_name == "validation":
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class))
            .batch(batch_size)
            .map(lambda img, img_class: (tf.py_function(augment, [img], [tf.float32])[0], img_class), num_parallel_calls=tf.data.AUTOTUNE,)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class),)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )  
    return(ds)
