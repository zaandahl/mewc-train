import os, sys, warnings
import tensorflow as tf
import pandas as pd

from keras_cv import layers
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Ensures the output directory exists
def ensure_output_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Processes the custom sample file
def process_custom_sample_file(custom_sample_file):
    is_custom_sample = len(custom_sample_file) > 1
    custom_sample_file_temp = {}

    for key, values in custom_sample_file.items():
        if key == 'default':
            custom_sample_file_temp[key] = values
        elif key == 'specific':
            for num_samples, class_list in values.items():
                for class_name in class_list:
                    custom_sample_file_temp[str(class_name)] = int(num_samples)

    return custom_sample_file_temp, is_custom_sample

# Validates the directory structure of the training and test paths
def validate_directory_structure(train_path, val_path, test_path):

# Checks if the upload format matches the train/validation/test format
    def check_upload_format(main_directory):
        # Check if main directory exists
        if not os.path.exists(main_directory):
            raise FileNotFoundError("Main directory does not exist.")
        
        # Check if it's a directory
        if not os.path.isdir(main_directory):
            raise NotADirectoryError("Path is not a directory.")
        
        # Get list of subdirectories
        subdirectories = [name for name in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, name))]
        
        # Check if each subdirectory contains only images
        for subdir in subdirectories:
            subdir_path = os.path.join(main_directory, subdir)
            files = os.listdir(subdir_path)
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    raise ValueError(f"File '{file}' in '{subdir}' is not a valid image file to train.")
        
        return True

    try:
        if check_upload_format(train_path) and check_upload_format(val_path) and check_upload_format(test_path):
            print("Directory structure is valid." + '\n')
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print("Error:", e)
        sys.exit(1)  # Exit training if the directory structure is invalid

# Prints the number of images in the dataset and the number of images in each class
def print_dsinfo(ds_df, ds_name='default'):
    print('Dataset: ' + ds_name)
    print(f'Number of images in the dataset: {ds_df.shape[0]}')
    print(str(ds_df['Label'].value_counts()) + '\n')
    
# Used to create a sampled train or validation/test sets when the latter comes from a different pool of images
def create_train(ds_path, seed=12345, ns=1000, custom_sample=False, custom_file=None):
    
    # Creates a sampled pandas dataframe with the image path and class label derived from the directory structure
    def create_dataframe(ds_path, n, seed):
        if custom_sample:
            dir_ = Path(ds_path)
            ds_filepaths = list(dir_.glob('**/*.[jJ][pP][gG]'))
            ds_labels = [os.path.split(os.path.split(x)[0])[1] for x in ds_filepaths]
            ds_filepaths = [str(x) for x in ds_filepaths]
            ds_df = pd.DataFrame({'File': ds_filepaths, 'Label': ds_labels})

            sampled_dfs = []
            for label in set(ds_labels):  # Loop through all unique labels in the dataset
                n_samples = int(custom_file.get(label, custom_file['default']))  # Get sample size from distribution or use default
                label_df = ds_df[ds_df['Label'] == label]
                if len(label_df) < n_samples:
                    sampled_df = label_df.sample(n=n_samples, replace=True, random_state=seed)
                else:
                    sampled_df = label_df.sample(n=n_samples, replace=False, random_state=seed)
                sampled_dfs.append(sampled_df)

            ds_df = pd.concat(sampled_dfs)
            ds_df = ds_df.sample(frac=1, random_state=seed).reset_index(drop=True)

            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)

            return ds_df
        else:    
            dir_ = Path(ds_path) # Selecting folder paths in dataset
            ds_filepaths = list(dir_.glob('**/*.[jJ][pP][gG]'))
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
    dir_ = Path(ds_path) # Selecting folder paths in dataset
    ds_filepaths = list(dir_.glob('**/*.[jJ][pP][gG]'))
    ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths)) # Mapping labels...
    ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str) # Data set paths & labels
    ds_labels = pd.Series(ds_labels, name='Label')
    ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
    return ds_df

# This function takes a pandas df from create_dataframe and converts to a TensorFlow dataset
def create_tensorset(in_df, img_size, batch_size, magnitude=0, n_augments=0, ds_name="test"):
    
    # helper function to use with the lambda mapping
    def load(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        img = tf.image.resize(img, size=(img_size, img_size))
        return img

    in_path = in_df['File']
    in_class = LabelEncoder().fit_transform(in_df['Label'].values)

    in_class = in_class.reshape(len(in_class), 1)
    in_class = OneHotEncoder(sparse_output=False).fit_transform(in_class)

    # If the magnitude is out of bounds, raise an error to notify
    if not (magnitude >= 0 and magnitude <= 1):
        magnitude = 0.1 # default to low value
        warnings.warn("Magnitude is out of bounds, default value set to 0.1", Warning)

    # using keras_cv random augmentation technique with user-selected no. augmentations per image and magnitude ranging [0,1]     
    rand_aug = layers.RandAugment(value_range=(0, 255), augmentations_per_image=n_augments, magnitude=magnitude, magnitude_stddev=(magnitude/3))

    # convert to dataset
    if ds_name == "train":
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class))
            .batch(batch_size)
            .map(lambda img, img_class: (rand_aug(tf.cast(img, tf.uint8)), img_class), num_parallel_calls=tf.data.AUTOTUNE,)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class),)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )  
    return(ds)
