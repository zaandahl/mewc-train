import os, sys, warnings
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
import pandas as pd
from keras_cv import layers
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def ensure_output_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_samples_from_config(config):
    custom_sample_file_temp = {}
    for entry in config["CLASS_SAMPLES_SPECIFIC"]:
        class_name = entry["CLASS"]
        num_samples = entry["SAMPLES"]
        custom_sample_file_temp[class_name] = num_samples
    is_custom_sample = True if custom_sample_file_temp else False
    return custom_sample_file_temp, is_custom_sample

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

def check_upload_format(main_directory):
    if not os.path.exists(main_directory):
        raise FileNotFoundError("Main directory does not exist.")
    if not os.path.isdir(main_directory):
        raise NotADirectoryError("Path is not a directory.")
    subdirectories = [name for name in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, name))]
    for subdir in subdirectories:
        subdir_path = os.path.join(main_directory, subdir)
        files = os.listdir(subdir_path)
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                raise ValueError(f"File '{file}' in '{subdir}' is not a valid image file to train.")    
    return True

def validate_directory_structure(train_path, val_path, test_path):
    try:
        if check_upload_format(train_path) and check_upload_format(val_path) and check_upload_format(test_path):
            print("Directory structure is valid." + '\n')
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print("Error:", e)
        sys.exit(1)

def print_dsinfo(ds_df, ds_name='default'):
    print('Dataset: ' + ds_name)
    print(f'Number of images in the dataset: {ds_df.shape[0]}')
    print(str(ds_df['Label'].value_counts()) + '\n')

def create_dataframe(ds_path, n, seed, custom_sample=False, custom_file=None):
    if custom_sample:
        dir_ = Path(ds_path)
        ds_filepaths = list(dir_.glob('**/*.[jJ][pP][gG]'))
        ds_labels = [os.path.split(os.path.split(x)[0])[1] for x in ds_filepaths]
        ds_filepaths = [str(x) for x in ds_filepaths]
        ds_df = pd.DataFrame({'File': ds_filepaths, 'Label': ds_labels})
        sampled_dfs = []
        for label in set(ds_labels):
            n_samples = int(custom_file.get(label, custom_file['default']))
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
        dir_ = Path(ds_path)
        ds_filepaths = list(dir_.glob('**/*.[jJ][pP][gG]'))
        ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths))
        ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str)
        ds_labels = pd.Series(ds_labels, name='Label')
        ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
        ds_df = ds_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        sampled_dfs = []
        for name, group in ds_df.groupby('Label'):
            sampled_dfs.append(group.sample(n=n, replace=len(group) < n, random_state=seed))
        ds_df = pd.concat(sampled_dfs).reset_index(drop=True)
        return ds_df

def create_train(ds_path, seed=12345, ns=1000, custom_sample=False, custom_file=None):
    train_df = create_dataframe(ds_path, ns, seed, custom_sample, custom_file)
    num_classes = train_df['Label'].nunique()
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return(train_df, num_classes)

def create_fixed(ds_path):
    dir_ = Path(ds_path)
    ds_filepaths = list(dir_.glob('**/*.[jJ][pP][gG]'))
    ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths))
    ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str)
    ds_labels = pd.Series(ds_labels, name='Label')
    ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
    return ds_df

def load_img(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    img = tf.image.resize(img, size=(img_size, img_size))
    return img

def create_tensorset(in_df, img_size, batch_size, magnitude=0, n_augments=0, ds_name="test"):
    in_path = in_df['File']
    in_class = LabelEncoder().fit_transform(in_df['Label'].values)
    in_class = in_class.reshape(len(in_class), 1)
    in_class = OneHotEncoder(sparse_output=False).fit_transform(in_class)
    if not (magnitude >= 0 and magnitude <= 1):
        magnitude = 0.1
        warnings.warn("Magnitude is out of bounds, default value set to 0.1", Warning)
    rand_aug = layers.RandAugment(value_range=(0, 255), augmentations_per_image=n_augments, magnitude=magnitude, magnitude_stddev=(magnitude/3))
    if ds_name == "train":
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load_img(img_path, img_size), img_class))
            .batch(batch_size)
            .map(lambda img, img_class: (rand_aug(tf.cast(img, tf.uint8)), img_class), num_parallel_calls=tf.data.AUTOTUNE,)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load_img(img_path, img_size), img_class),)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )  
    return(ds)
