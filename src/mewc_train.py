# Fit classifier with frozen base model then fine-tune progressively
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
import tensorflow as tf
from tqdm import tqdm
from lib_data import read_yaml, print_dsinfo, create_train, create_fixed, create_tensorset
from lib_model import build_classifier, fit_frozen, fit_progressive, exp_scheduler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

config = read_yaml("config.yaml") 
for conf_key in config.keys():
    if conf_key in os.environ:
        config[conf_key] = os.environ[conf_key]

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

train_df, num_classes = create_train(
    config['TRAIN_PATH'],
    seed=config['SEED'],
    ns=config['N_SAMPLES']
)
print_dsinfo(train_df, 'Training Data')
print('Number of classes: {}'.format(num_classes))

val_df = create_fixed(config['TEST_PATH']) # create a fixed validation set (also used for creating the non-augmented test set)
print_dsinfo(val_df, 'Validation Data')

test_df = create_tensorset(
        val_df.sample(frac=1).reset_index(drop=True), 
        img_size=config['SHAPES'][0], 
        batch_size=config['BATCH_SIZES'][0], 
        magnitude=0, # no augmentation on test set
        ds_name="test" # set to test to turn off augmentation, or train or validation to include it
)

if not os.path.exists(config['OUTPUT_PATH']):
    os.makedirs(config['OUTPUT_PATH'])

with strategy.scope(): # This first trains the DenseNet(s), but leaves the base model frozen, to avoid catastrophic cascades into the base layers
    en_model = build_classifier(nc=num_classes, mod=config['MODEL'], size=config['SHAPES'][0], compression=config['CLW'], lr=5e-5, dr=0.1) # use high LR and low DR on frozen model

    frozen_hist, en_model = fit_frozen(
        en_model,
        num_classes,
        epochs=config['FROZ_EPOCH'],
        target_shape=config['SHAPES'][0],
        dropout=config['DROPOUTS'][0],
        magnitude=config['MAGNITUDES'][0],
        batch_size=config['BATCH_SIZES'][0],
        output_path=config['OUTPUT_PATH'],
        train_df=train_df,
        val_df=val_df,
        layers_to_unfreeze=config['LUF'] # -1 unfreezes the whole model, 0 freezes base model, select 1 or greater to unfreeze n layers (blocks, see config comments)
)

print("Total trainable base-model layers: {}".format(len(en_model.layers[0].trainable_weights))) 
print("Frozen model: test loss, test acc:", en_model.evaluate(test_df, batch_size=config['BATCH_SIZES'][0]))
en_model.summary() # print model summary

prog_train = [] # store training samples for each epoch
prog_val = []  # store validation samples for each epoch
for i in tqdm(range(config['PROG_TOT_EPOCH'])):
    train_tmp, num_classes = create_train(
        config['TRAIN_PATH'],
        seed=config['SEED'] + i * config['SEED'],
        ns=config['N_SAMPLES']
    )
    prog_train.append(train_tmp)
    prog_val.append(val_df) # validation images remain constant (but are augmented each progress iteration)

prog_hists, en_model = fit_progressive(
        en_model,
        prog_train,
        prog_val,
        test_df,
        savefile=config['SAVEFILE'],
        output_path=config['OUTPUT_PATH'],
        lr_scheduler=exp_scheduler,
        total_epochs=config['PROG_TOT_EPOCH'],
        prog_stage_len=config['PROG_STAGE_LEN'],
        magnitudes=config['MAGNITUDES'],
        dropouts=config['DROPOUTS'],
        target_shapes=config['SHAPES'],    
        batch_sizes=config['BATCH_SIZES']
    )

print("Final Epoch: test loss, test acc:", en_model.evaluate(test_df, batch_size=config['BATCH_SIZES'][0]))
