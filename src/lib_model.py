import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

from tensorflow import keras
from keras.models import Sequential, load_model
from keras import layers
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.applications import mobilenet_v3, efficientnet_v2

from lib_data import create_tensorset

from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import rcParams

def exp_scheduler(epoch, lr):  
    lr_base=5e-5; decay_step=5; decay_rate=0.9; lr_min=1e-7; warmup=1; alpha=2 # lr_base=5e-5 as default, alpha for exp decay tuned to a 10-epoch warmup

    if epoch < warmup:
        lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1) # this gives linear warmup
        #lr = lr_min + lr_min * epoch ** alpha # this yields a slow warmup with exponential growth

    else:
        lr = lr_base * decay_rate ** ((epoch - warmup) / decay_step)
        lr = lr if lr > lr_min else lr_min
    return lr

def build_classifier(nc, mod, size, compression, lr, dr):
    if nc == 2:
        loss_f = 'binary_crossentropy'
        act_f = 'sigmoid'
    else:
        loss_f = tfa.losses.SigmoidFocalCrossEntropy() # use tfa.losses.SigmoidFocalCrossEntropy()
        act_f = 'softmax'
    
    if mod == 'MN-V3-S':
        model_base = mobilenet_v3.MobileNetV3Small(include_top=False, input_shape=(size,size,3), pooling="avg", weights='imagenet')     #  1.0M, 224px
    elif mod == 'EN-B0':
        model_base = efficientnet_v2.EfficientNetV2B0(include_top=False, input_shape=(size,size,3), pooling="avg", weights='imagenet')  #  6.0M, 224px
    elif mod == 'EN-V2S':    
        model_base = efficientnet_v2.EfficientNetV2S(include_top=False, input_shape=(size,size,3), pooling="avg", weights='imagenet')  #  20.4M, 300px    
    elif mod == 'EN-V2M':    
        model_base = efficientnet_v2.EfficientNetV2M(include_top=False, input_shape=(size,size,3), pooling="avg", weights='imagenet')   # 53.2M, 384px
    elif mod == 'EN-V2L':    
        model_base = efficientnet_v2.EfficientNetV2L(include_top=False, input_shape=(size,size,3), pooling="avg", weights='imagenet')  # 117.8M, 480px 
    else:
        model_base = load_model(mod)
        model_base = model_base.layers[0]

    model_base.trainable = False # Freeze the pretrained weights (to put the model in inference mode)
    model = Sequential() # create a sequential model
    model.add(model_base) # add the base model to the sequential model
         
    model.add(Dropout(dr, name="base_dropout")) # reqularization layer (random dropouts of a proportion of all neurons) 
    model.add(Dense(compression, activation='relu', name="compression_bottleneck")) 
    model.add(Dropout(dr, name="top_dropout")) # reqularization layer (random dropouts of a proportion of all neurons) 
    model.add(Dense(nc, activation=act_f, name="classification")) # output layer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # optimizer, lr is the learning rate
    model.compile(optimizer=optimizer, loss=loss_f, metrics=["accuracy"]) # compile the model

    return model

def unfreeze_model(model, lr, num_classes, layers_to_unfreeze):
    if num_classes == 2:
        loss_f = 'binary_crossentropy'
    else:
        loss_f = tfa.losses.SigmoidFocalCrossEntropy() # use tfa.losses.SigmoidFocalCrossEntropy() or polyl_cross_entropy

    # Both the whole model and specific layers need to be trainable
    # So make whole model trainable, then aftewards freeze all layers you don't want to train
    for layer in model.layers[-10:]: # This unfreezes the whole model since it is stored in layer 0, -10 is safe even with multiple dense top
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True    
    
    mlen = len(model.layers[0].layers) # number of layers in the base model of the keras sequential model stack

    if layers_to_unfreeze == -1 :
        for layer in range(1, mlen):
            ml = model.layers[0].layers[layer]
            if isinstance(ml, layers.BatchNormalization): # Freezes all BatchNormalization layers in the convnet base
                ml.trainable = False

    elif layers_to_unfreeze > 0 :
        # We need to freeze all layers except those we wish to fine-tune, if LUF is specified
        mlen = len(model.layers[0].layers) # number of layers in the base model of the keras sequential model stack
        for layer in range(1, mlen):
            ml = model.layers[0].layers[layer]
            if isinstance(ml, layers.BatchNormalization) or layer<(mlen-layers_to_unfreeze): # freeze all BatchNorm and layers lower than LUF
                ml.trainable = False
    
    else:
        model.layers[0].trainable = False # Freeze the pretrained weights of the base model

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_f, metrics=["accuracy"])
    return(model)

def fit_frozen(
    model,
    num_classes,
    train_df,
    val_df,
    output_path,
    epochs=50,
    batch_size=64,
    target_shape=224,
    dropout=0.1,
    dropout_layer=-3,
    magnitude=5,
    out_lr=5e-6,
    layers_to_unfreeze=-1
):
    cb = [EarlyStopping(monitor='loss', mode='min', min_delta=0.001, patience=5, restore_best_weights=True)]
    train_ds = create_tensorset(train_df, img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="train")
    val_ds = create_tensorset(val_df, img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="validation")
    
    if dropout != None and isinstance(model.layers[dropout_layer], keras.layers.Dropout):
        print(">>>> Changing dropout rate to:", dropout)
        model.layers[dropout_layer].rate = dropout
    
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
    
    model.save(output_path+'/frozen.h5', save_format="h5")
    unfreeze_model(model, lr=out_lr, num_classes=num_classes, layers_to_unfreeze=layers_to_unfreeze)
    model.save(output_path+'/unfrozen.h5', save_format="h5")

    return(hist, model)

def fit_progressive(
    model,
    train_df,
    val_df,
    test_ds,
    savefile,
    output_path,
    lr_scheduler=None,
    total_epochs=50,
    prog_stage_len=10,
    dropout_layer=-3,
    magnitudes=[1],
    dropouts=[0.01],
    target_shapes=[224],
    batch_sizes=[64]
):
    if model.compiled_loss is None:
        print(">>>> Error: Model NOT compiled.")
        return None
    
    histories = []
    lowest_loss = 1e6 # monitor validation loss (initialise)
    stages = len(batch_sizes)

    for stage, batch_size, target_shape, dropout, magnitude in zip(range(stages), batch_sizes, target_shapes, dropouts, magnitudes):
        if stage == 0:
            seq = range(stage, prog_stage_len)
        elif stage < (stages-1):  
            seq = range(stage*prog_stage_len, (stage*prog_stage_len)+prog_stage_len)
        else:
            seq = range(stage*prog_stage_len, total_epochs)

        print("") 
        print(">>>> Stage: {}/{}, target_shape: {}, dropout: {}, magnitude: {}, batch_size: {}".format(stage+1, stages, target_shape, dropout, magnitude, batch_size)) 
        print(">>>> This stage runs from epoch {} to {} out of {} total epochs".format(seq[0]+1, seq[len(seq)-1]+1, total_epochs))

        if len(dropouts) > 1 and isinstance(model.layers[dropout_layer], keras.layers.Dropout):
            model.layers[dropout_layer].rate = dropout
 
        for i in seq:
            train_ds = create_tensorset(train_df[i], img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="train")
            val_ds = create_tensorset(val_df[i], img_size=target_shape, batch_size=batch_size, magnitude=magnitude, ds_name="validation")
            
            history = model.fit(
                train_ds,
                initial_epoch=i,
                epochs=(i+1),  
                validation_data=val_ds,
                callbacks=[LearningRateScheduler(lr_scheduler)] if lr_scheduler is not None else [],
            )        
            histories.append(history)

            epoch_test_loss = float(model.evaluate(test_ds, batch_size=batch_size)[0]) # check on non-augmented test data to see if best epoch
            if epoch_test_loss <= lowest_loss: # save model only if current validation loss is as good as or better than all previous epochs
                lowest_loss = epoch_test_loss # update new best val_acc
                best_model_fpath = output_path+'/'+savefile+'_'+str(target_shape)+'px_best.h5'
                model.save(best_model_fpath, save_format="h5")
                print('New best-performing epoch of model (size = {}px) saved as: {}'.format(target_shape, best_model_fpath))

    model.save(output_path+'/'+savefile+'_'+str(target_shape)+'px_final.h5', save_format="h5") # save final epoch as well
    print('Final model at epoch {} of model (size = {}px) saved as: {}'.format(total_epochs, target_shape, output_path+'/'+savefile+'_'+str(target_shape)+'px_final'))
    hhs = {kk: np.ravel([hh.history[kk] for hh in histories]).astype("float").tolist() for kk in history.history.keys()}
    return(hhs, model, best_model_fpath)

# evaluate and calculate class-specific metrics
def calc_class_metrics(
    model_fpath,
    classes,
    batch_size,
    image_size,
    output_path
):
    # load model and classes
    print(f"Calculating class-specific metrics for best model '{model_fpath}'")
    loaded_model = load_model(model_fpath, custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})
    class_map = {name: idx for idx, name in enumerate(classes)}
    inv_class = {v: k for k, v in class_map.items()}
    class_ids = sorted(inv_class.values())

    # make predictions
    y_pred = []
    y_true = []
    for i, spp_class in enumerate(classes):
        print(f"\nEvaluating class '{spp_class}' ({(i+1)}/ {len(classes)})")
        img_generator = tf.keras.preprocessing.image_dataset_from_directory(
            "/data/test/" + spp_class, 
            labels=None,
            label_mode=None,
            batch_size=int(batch_size), 
            image_size=(int(image_size), int(image_size)),
            shuffle=False)
        preds = loaded_model.predict(img_generator)
        y_pred_tmp = [class_ids[pred.argmax()] for pred in preds]
        y_pred.extend(y_pred_tmp)
        y_true.extend([spp_class] * len(y_pred_tmp))

    # calculate metrics
    print("\n\nClassification report:")
    print(metrics.classification_report(y_true, y_pred))

    # plot confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, normalize = "true")
    rcParams.update({'figure.autolayout': True})
    nc = len(class_ids)
    if nc > 20: # adjust font size with many classes
        font_size = 7 if nc < 35 else 5
        rcParams.update({'font.size': font_size})
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_ids)
    cm_display.plot(cmap=plt.cm.Blues, include_values = len(class_ids) < 8, values_format = '.2g') # only include values with few classes
    plt.xticks(rotation=90, ha='center')
    plt.savefig(output_path + "/confusion_matrix.png")
