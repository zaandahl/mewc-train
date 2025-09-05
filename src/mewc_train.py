# Fit classifier with frozen base model then fine-tune progressively
import os, yaml
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["KERAS_BACKEND"] = "jax"
import tensorflow as tf
print("\nTensorFlow version:", tf.__version__)

from tqdm import tqdm # for progress bar
from lib_common import update_config_from_env, model_img_size_mapping, read_yaml, setup_strategy, get_mod
from lib_model import build_classifier, fit_frozen, fit_progressive, calc_class_metrics
from lib_data import print_dsinfo, create_train, create_fixed, process_samples_from_config, ensure_output_directory, validate_directory_structure

config = update_config_from_env(read_yaml("config.yaml"))
custom_sample_file, is_custom_sample = process_samples_from_config(config)

strategy = setup_strategy() # Set up the strategy for distributed training
output_fpath = os.path.join(config['OUTPUT_PATH'], config['SAVEFILE'], get_mod(config['MODEL']))
ensure_output_directory(output_fpath)
validate_directory_structure(config['TRAIN_PATH'], config['VAL_PATH'], config['TEST_PATH'])

img_size = model_img_size_mapping(get_mod(config['MODEL']))

train_df, num_classes = create_train(
    config['TRAIN_PATH'],
    seed=config['SEED'],
    ns=custom_sample_file['default'], custom_sample=is_custom_sample, custom_file=custom_sample_file
)
classes = train_df['Label'].unique()
class_map = {name: idx for idx, name in enumerate(classes)}
df_size = train_df.shape[0]

with open(os.path.join(output_fpath, config['SAVEFILE']+'_class_map.yaml'), 'w') as file:
    yaml.dump(class_map, file, default_flow_style=False)

print('Number of classes: {}'.format(num_classes) + '\n')
print_dsinfo(train_df, 'Training Data')

val_df = create_fixed(config['VAL_PATH']) # No image augmentation and fixed structure, for validation
print_dsinfo(val_df, 'Validation Data')

with strategy.scope(): # Trains the Dense classifier, but leaves the base model frozen, to avoid catastrophic cascades
    model = build_classifier(config, num_classes, df_size, img_size)
    frozen_hist, model = fit_frozen(config, model, train_df, val_df, num_classes, df_size, img_size)
model.summary() # print model summary

print('\nCreating datasets for progressive training...')
prog_train = [] # store training data samples for each epoch
for i in tqdm(range(config['PROG_TOT_EPOCH'])):
    train_tmp, num_classes = create_train(
        config['TRAIN_PATH'],
        seed=(config['SEED']+i), # change seed for each progress iteration
        ns=custom_sample_file['default'], custom_sample=is_custom_sample, custom_file=custom_sample_file
    )
    prog_train.append(train_tmp)

prog_hists, model, best_model_fpath = fit_progressive(
        config, model,
        train_df = prog_train,
        val_df = val_df,
        output_fpath = output_fpath,
        img_size = img_size
    )

calc_class_metrics(
    model_fpath = best_model_fpath,
    test_fpath = config['TEST_PATH'],
    output_fpath = output_fpath,
    classes = classes,
    batch_size = config["BATCH_SIZE"],
    img_size = img_size
    )
