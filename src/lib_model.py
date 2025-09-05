import os, sys, kimm
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks, layers, losses, models, optimizers, utils, saving
from huggingface_hub import hf_hub_download
from lib_data import create_tensorset
from sklearn import metrics
from matplotlib import rcParams
from lib_common import get_mod

# Configures the optimizer and loss function
def configure_optimizer_and_loss(config, num_classes, df_size):
    # Determine loss function and activation function based on number of classes
    if num_classes == 2:
        loss_f = losses.BinaryFocalCrossentropy()  # Use for binary classification tasks
        act_f = 'sigmoid' # Use for binary classification tasks
    else:
        loss_f = losses.CategoricalFocalCrossentropy()  # Use for unbalanced multi-class tasks (typical for wildlife datasets)
        act_f = 'softmax' # Use for multi-class classification tasks
    lr_scheduler = config['LR_SCHEDULER'] # Learning rate scheduler
    optim = config['OPTIMIZER'] # Gradient-based optimisation algorithm
    weight_decay = float(config['OPTIM_REG']) # Weight decay regularisation for optimiser
    lr_init = float(config['LEARNING_RATE']) # Initial learning rate for the scheduler
    if optim == 'lion': lr_init = lr_init / 5 # Decrease learning rate for Lion optimiser due to use of sign operator 
    # Determine learning rate schedule settings
    min_lr_frac = 1/5 # Default minimum learning rate fraction of initial learning rate
    steps_per_epoch = df_size // config['BATCH_SIZE'] # Number of batches per epoch
    decay_steps = int(config['PROG_STAGE_LEN'] * steps_per_epoch) # Cycling the LR based on progressive stage length
    if lr_scheduler == 'polynomial': # Polynomial decay with linear, quadratic, sqrt or cubic decay
        end_learning_rate = lr_init * min_lr_frac # lowest allowed learning rate
        power = 1 # default is linear = 1 (2 for quadratic, 0.5 for sqrt, 3 for cubic)
        lr = optimizers.schedules.PolynomialDecay(initial_learning_rate=lr_init, decay_steps=decay_steps*2, end_learning_rate=end_learning_rate, power=power, cycle=True)
    elif lr_scheduler == 'cosine': # Cosine decay with warm restarts for cyclical learning rates
        t_mul = 1.3  # Multiplier for increasing the number of iterations in the i-th period
        m_mul = 3/4  # Multiplier for changing the initial learning rate of the i-th period
        lr = optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr_init, first_decay_steps=decay_steps, t_mul=t_mul, m_mul=m_mul, alpha=min_lr_frac)
    else: # Default to exponential decay with a fixed minimum learning rate fraction
        total_steps = config['PROG_TOT_EPOCH'] * steps_per_epoch # Total number of steps for monotonic exponential decay across all epochs
        lr = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_init, decay_steps=total_steps, decay_rate=min_lr_frac, staircase=False)
    # Determine optimiser settings
    if optim == 'rmsprop': # Simple and well-known RMSprop optimizer with default settings and centered gradients
        optimizer = optimizers.RMSprop(learning_rate=lr, rho=0.9, centered=True, weight_decay=weight_decay)
    elif optim == 'lion': # New efficient optimiser with sign operator for gradient updates and weight decay regularisation
        weight_decay = weight_decay * 5 # Increased weight decay for Lion optimiser relative to AdamW to counter decreased LR due to sign operator
        optimizer = optimizers.Lion(learning_rate=lr, weight_decay=weight_decay)
    else: # Default to AdamW, a robust momentum-based optimiser with AMSGrad smoothing and weight-decay regularisation
        optimizer = optimizers.AdamW(learning_rate=lr, amsgrad=True, weight_decay=weight_decay)
    return optimizer, loss_f, act_f

# Import base model from kimm or huggingface or local file
def import_model(img_size, mname, REPO_ID, FILENAME):
    # Dictionary mapping mod strings to model constructors
    # print(kimm.list_models(weights="imagenet")) # to check all available kimm models
    model_constructors = {
        'en0': kimm.models.EfficientNetV2B0, # 5 M model params : 26 MB frozen file size
        'en2': kimm.models.EfficientNetV2B2, # 9 M : 37 MB
        'ens': kimm.models.EfficientNetV2S, # 21 M : 84 MB
        'enm': kimm.models.EfficientNetV2M, # 54 M : 216 MB
        'enl': kimm.models.EfficientNetV2L, # 119 M : 475 MB
        'enx': kimm.models.EfficientNetV2XL, # 208 M : 835 MB
        'cnp': kimm.models.ConvNeXtPico, # 9 M : 35 MB
        'cnn': kimm.models.ConvNeXtNano, # 16 M : 61 MB
        'cnt': kimm.models.ConvNeXtTiny, # 29 M : 112 MB
        'cns': kimm.models.ConvNeXtSmall, # 50 M : 200 MB 
        'cnb': kimm.models.ConvNeXtBase, # 89 M : 352 MB
        'cnl': kimm.models.ConvNeXtLarge, # 198 M : 787 MB
        'vtt': kimm.models.VisionTransformerTiny16, # 6 M : 23 MB
        'vts': kimm.models.VisionTransformerSmall16, # 22 M : 88 MB
        'vtb': kimm.models.VisionTransformerBase16, # 87 M : 346 MB 
        'vtl': kimm.models.VisionTransformerLarge16 # 305 M : 1.22 GB
    # Note: actual trained model file size (MB/GB) will depend on number of blocks unfrozen during fine-tuning
    }
    try:
        filepath = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        model_base = models.load_model(filepath, custom_objects=None, compile=False, safe_mode=True)
        model_base = model_base.layers[0] # extract the base model from the keras sequential model stack (removing the custom top layers)
        print(f"Weights from pretrained model \"{filepath}\" successfully loaded")
    except:
        print("No HuggingFace pretrained model to download: using ImageNet weights") 
        if mname.lower() in model_constructors:
            model_base = model_constructors[mname.lower()](include_top=False, input_shape=[img_size, img_size, 3])
        else:
            print("Loading custom pretrained model from local file")
            model_base = models.load_model(mname, compile=False) # load the local filename specified in config.yaml
            model_base = model_base.layers[0] # extract the base model from the keras sequential model stack (removing the custom top layers)
    return model_base

# Configure and add model top to the base model
def build_sequential_model(model_base, num_classes, act_f, mname, dr):
    model_base.trainable = False # Freeze the pretrained weights (to put the model in inference mode)
    model = models.Sequential() # create a sequential model
    model.add(layers.Input(shape=model_base.input_shape[1:])) # Add an Input layer with the correct input shape   
    model.add(model_base) # add the base model to the sequential model 
    # Conditionally add the 1D global average pooling layer if mod contains "VT", else use 2D:
    if "vt" in mname[:2].lower():
        model.add(layers.GlobalAveragePooling1D(name="global_average_pooling"))
    else:
        model.add(layers.GlobalAveragePooling2D(name="global_average_pooling"))
    model.add(layers.Dropout(dr, name="base_dropout")) # reqularization layer (random dropouts of a proportion of all neurons)      
    # Conditionally calculate the compression layer depending on whether mod contains "VT" or on the number of classes:
    if "vt" in mname[:2].lower():
        compression_layer = (
            128 if model_base.output_shape[2] <= 384 
            else 256 if num_classes <= 25 
            else 384 if model_base.output_shape[2] == 768 
            else 512
        )
    else:
        compression_layer = (
            256 if num_classes <= 25 or model_base.output_shape[3] < 768 
            else 384 if model_base.output_shape[3] == 768 
            else 512
        )
    model.add(layers.Dense(compression_layer, name="compression_bottleneck")) 
    model.add(layers.ELU(alpha=1.0)) # Exponential Linear Unit activation function to avoid dead neurons 
    model.add(layers.Dropout(dr, name="top_dropout")) # second dropout regularization layer before the dense classifier     
    model.add(layers.Dense(num_classes, activation=act_f, name="classification")) # output layer
    return model

# Builds the classifier model
def build_classifier(config, num_classes, df_size, img_size):
    optimizer, loss_f, act_f = configure_optimizer_and_loss(config, num_classes, df_size)
    model_base = import_model(img_size, mname=config['MODEL'], REPO_ID = config['REPO_ID'], FILENAME = config["FILENAME"])
    model = build_sequential_model(model_base, num_classes, act_f, mname=get_mod(config['MODEL']), dr=config['DROPOUTS'][0])
    model.compile(optimizer=optimizer, loss=loss_f, metrics=["accuracy"])
    print('Model built and compiled\n')
    return model

# Finds the unfreeze points in the model for fine-tuning
def find_unfreeze_points(model, mname, blocks_to_unfreeze):
    block_starts = []
    print("\nModel Name =", mname)
    if 'cn' in mname[:2].lower():
        for layer in model.layers:
            if 'stages' in layer.name and layer.name.endswith('_downsample_1_conv2d'):
                block_starts.append(layer.name)
    elif 'en' in mname[:2].lower():
        for layer in model.layers:
            if 'block' in layer.name and layer.name.endswith('_0_conv_pw_conv2d'):
                block_starts.append(layer.name)
    elif 'vt' in mname[:2].lower():
        for layer in model.layers:
            if 'blocks' in layer.name and layer.name.endswith('_attn'):
                block_starts.append(layer.name)
    if len(block_starts) < blocks_to_unfreeze:
        raise ValueError("Number of blocks to unfreeze exceeds available blocks.")
    return block_starts[-blocks_to_unfreeze:] if blocks_to_unfreeze > 0 else []

# Unfreezes blocks of the model for fine-tuning
def unfreeze_model(config, model, num_classes, df_size):
    optimizer, loss_f, act_f = configure_optimizer_and_loss(config, num_classes, df_size)
    model.layers[0].trainable = True # Set whole model as trainable by default, then selectively freeze layers
    blocks_to_unfreeze=config['BUF']
    if blocks_to_unfreeze == 0:
        print("Freezing all layers for fine-tuning of model classifier only")
        model.layers[0].trainable = False # Set whole base model as frozen 
    elif blocks_to_unfreeze > 0:
        unfreeze_points = find_unfreeze_points(model.layers[0], get_mod(config['MODEL']), blocks_to_unfreeze)
        print("Unfreezing blocks:", unfreeze_points, "for constrained fine-tuning")
        if unfreeze_points:
            start_unfreezing = False
            for layer in model.layers[0].layers:
                if not start_unfreezing: # Freeze all base model layers below the first unfreeze layer
                    if layer.name == unfreeze_points[0]:
                        start_unfreezing = True # Start unfreezing layers from this point, except for BatchNorm layers
                        layer.trainable = True
                    else:
                        layer.trainable = False
                else:
                    if isinstance(layer, layers.BatchNormalization):
                        layer.trainable = False
                    else:
                        layer.trainable = True
    else:
        print("Unfreezing all layers for fine-tuning all model weights")
        for layer in model.layers[0].layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False # Freeze only the BatchNorm layers, as these should never be trainable
    model.compile(optimizer=optimizer, loss=loss_f, metrics=["accuracy"])
    return(model)

# Trains the model with the base model frozen to stabilise the classifier
def fit_frozen(config, model, train_df, val_df, num_classes, df_size, img_size):
    cb = [callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.001, patience=5, restore_best_weights=True)]
    train_ds = create_tensorset(train_df, img_size, batch_size=config['BATCH_SIZE']) # no augmentation for frozen model
    val_ds = create_tensorset(val_df, img_size, batch_size=config['BATCH_SIZE'])  
    hist = model.fit(train_ds, validation_data=val_ds, epochs=config['FROZ_EPOCH'], callbacks=cb)
    unfreeze_model(config, model, num_classes, df_size)
    return(hist, model)  
        
# Fine-tunes the model progressively with blocks unfrozen and increasing regularisation
def fit_progressive(config, model, train_df, val_df, output_fpath, img_size): 
    if model._compile_loss is None:
        print(">>> Error: Model NOT compiled.")
        sys.exit(1)
    histories = []
    lowest_loss = 1 # monitor validation loss (initialise at high value to ensure first epoch is saved as best model)
    dropout_layer=[-2,-4], # keep fixed unless top of model architecture is changed 
    stages = len(config['MAGNITUDES'])
    prog_stage_len = config['PROG_STAGE_LEN']
    total_epochs = max(config['PROG_TOT_EPOCH'], stages*prog_stage_len)
    best_model_fpath = os.path.join(output_fpath, config['SAVEFILE']+'_'+get_mod(config['MODEL'])+'.keras')
    val_ds = create_tensorset(val_df, img_size, batch_size=config['BATCH_SIZE'])
    for stage, dropout, magnitude in zip(range(stages), config['DROPOUTS'], config['MAGNITUDES']):
        if stage == 0:
            seq = range(stage, prog_stage_len)
        elif stage < (stages-1):  
            seq = range(stage*prog_stage_len, (stage*prog_stage_len)+prog_stage_len)
        else:
            seq = range(stage*prog_stage_len, total_epochs)        
        print("\n>>> Stage: {}/{}, dropout: {}, magnitude: {}".format(stage+1, stages, dropout, magnitude))
        print(">>> This stage runs from epoch {} to {} out of {} total epochs".format(seq[0]+1, seq[len(seq)-1]+1, total_epochs))
        if len(dropout_layer) > 1:
            for layer_index in dropout_layer:
                if isinstance(model.layers[layer_index], layers.Dropout):
                    model.layers[layer_index].rate = dropout 
        for i in seq:
            train_ds = create_tensorset(train_df[i], img_size, batch_size=config['BATCH_SIZE'], 
                                        magnitude=magnitude, n_augments=config['NUM_AUG'], ds_name="train")            
            history = model.fit(train_ds, initial_epoch=i, epochs=(i+1), validation_data=val_ds) 
            histories.append(history)
            epoch_val_loss = history.history['val_loss'][-1] # check on non-augmented validation data to see if best epoch
            if epoch_val_loss < lowest_loss: # save model only if current val loss is better than any previous epoch
                lowest_loss = epoch_val_loss
                saving.save_model(model, best_model_fpath, include_optimizer=False)
                print('New best-performing model epoch saved as: {}'.format(best_model_fpath))
    hhs = {kk: np.ravel([hh.history[kk] for hh in histories]).astype("float").tolist() for kk in history.history.keys()}
    return(hhs, model, best_model_fpath)

# Calculates class-specific metrics for the best model
def calc_class_metrics(model_fpath, test_fpath, output_fpath, classes, batch_size, img_size):
    nc = len(classes)
    print(f"\nCalculating class-specific metrics for best model '{model_fpath}'")
    loaded_model = models.load_model(model_fpath, compile=False)
    loaded_model.trainable = False # Freeze the whole model for inference-only mode thereafter
    saving.save_model(loaded_model, model_fpath, include_optimizer=False) # save best model in a frozen state for smaller file size    
    class_map = {name: idx for idx, name in enumerate(classes)}
    inv_class = {v: k for k, v in class_map.items()}
    class_ids = sorted(inv_class.values())
    y_pred = []
    y_true = []
    for i, spp_class in enumerate(classes):
        print(f"\nEvaluating class '{spp_class}' ({(i+1)}/ {len(classes)})")
        img_generator = utils.image_dataset_from_directory(
            test_fpath + "/"  + spp_class,
            labels=None,
            label_mode=None,
            batch_size=batch_size, 
            image_size=(img_size, img_size),
            shuffle=False)
        preds = loaded_model.predict(img_generator)
        y_pred_tmp = [class_ids[pred.argmax()] for pred in preds]
        y_pred.extend(y_pred_tmp)
        y_true.extend([spp_class] * len(y_pred_tmp))
    print("\n\nClassification report:")
    print(metrics.classification_report(y_true, y_pred, digits=3))
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, normalize = "true")
    rcParams.update({'figure.autolayout': True})
    if nc > 20: # adjust font size with many classes
        font_size = 7 if nc < 35 else 5
        rcParams.update({'font.size': font_size})
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_ids)
    cm_display.plot(cmap=plt.cm.Blues, include_values = len(class_ids) < 8, values_format = '.2g') # only include values with few classes
    plt.xticks(rotation=90, ha='center')
    plt.savefig(os.path.join(output_fpath, "confusion_matrix.png"))
