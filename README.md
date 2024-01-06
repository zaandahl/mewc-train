<img src="mewc_logo_hex.png" alt="MEWC Hex Sticker" width="200" align="right"/>

# mewc-train

## Introduction
This repository contains code to build a Docker container for running mewc-train. This is a tool used to train a model for predicting species from camera trap images. The classifier engine used in mewc-train is EfficientNetV2. 

You can supply arguments via an environment file where the contents of that file are in the following format with one entry per line:
```
VARIABLE=VALUE
```

## Usage

After installing Docker you can run the container using a command similar to the following. The `--env CUDA_VISIBLE_DEVICES=0` and `--gpus all` options allow you to take advantage of GPU accelerated training if your hardware supports it. Substitute `"$DATA_DIR"` for your training data directory and create a text file `"$ENV_FILE"` with any config options you wish to override. 

The default structure under the data directory is as follows:
```
data
├── train
│   ├── class1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── test
    ├── class1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

The train data directory must contain at least one subdirectory with images for each class. The test data directory should be structured the same way and contain images for testing the model after training.

```
docker pull zaandahl/mewc-train
docker run --env CUDA_VISIBLE_DEVICES=0 --gpus all \ 
    --env-file "$ENV_FILE" \
    --interactive --tty --rm \
    --volume "$DATA_DIR":/data \
    zaandahl/mewc-train
```

## Model Outputs and Integration with mewc-predict
Upon successful completion of training with mewc-train, two primary outputs will be generated:

1. Trained Model (`mewc_model.h5`): This is the serialized version of your trained neural network and contains all the learned weights and biases. This file is crucial for making predictions on new, unseen data.

2. Class List (`class_list.yaml`): A YAML file that provides a mapping between class names and class IDs. This is especially vital to ensure that predictions made by the model are correctly associated with their respective class names.

When using `mewc-predict`, it expects both `mewc_model.h5` and `class_list.yaml` to be available. This ensures seamless predictions and accurate class labeling. Ensure you maintain the integrity of these files and store them securely to make the most out of your trained model.

## Config Options

The following environment variables are supported for configuration (and their default values are shown). Simply omit any variables you don't need to change and if you want to just use all defaults you can leave `--env-file $ENV_FILE` out of the command alltogether. 

The main volume mount in the docker command above maps your local data directory to the `/data` directory in the container. The default values below assume you have a directory structure as shown above. Remember that all paths are relative inside of the Docker container so `/data` exists in the container and is not a local path on your machine.

Output from the container will be saved in the `/data/output` directory. The class list derived from the train test directory structure will be saved to `class_list.yaml` as described in the section above. The output model file will be saved as `mewc_model.h5`. Additionally the container will save a frozen and unfrozen version of the model after initially training upon ImageNet. These files are simply saved as `frozen.h5` and `unfrozen.h5` and are used to initialize the progressive training stages. The best performing model after each progressive stage is saved as `mewc_model_224px_best.h5` where `224px` is the image size used for that stage.

Note that for the `MAGNITUDES`, `DROPOUTS`, `SHAPES` and `BATCH_SIZES` variables you can supply multiple values separated by commas. The values will be used in sequence for each progressive training stage. For example, if you supply `MAGNITUDES=5,15,25` then the first stage will use a magnitude of 5, the second stage will use 15 and the third stage will use 25. The length of these four variables must be the same. For example in the default values shown below there are three values for each variable. 

| Variable | Default | Description |
| ---------|---------|------------ |
| SEED | 12345 | random seed for reproducibility |
| MODEL | EN-B0 | MN-V3-S, EN-B0, EN-V2S, EN-V2M, EN-V2L, or else supply base-model filename |
| CLW | 256 | width of the compression bottleneck layer (MN-V3-5 = 128, BO = 256, 512 for others) |
| LUF | 193 | Layers to Unfreeze: MN-V3-S = 53, B0 = 193, EN-V2S = 360, EN-V2M = 345, EN-V2L = 480 |
| SAVEFILE | mewc_model | filename to save model |
| CLASSLIST | class_list.yaml | filename to save class list |
| TRAIN_PATH | /data/train | path to training data |
| TEST_PATH | /data/test | path to test data |
| OUTPUT_PATH | /data/output | path to save output |
| N_SAMPLES | 4000 | number of samples to use for training per class |
| FROZ_EPOCH | 15 | number of epochs to train frozen model to converge the dense classifier |
| PROG_STAGE_LEN | 10 | progressive number of epochs prior to final sequence |
| PROG_TOT_EPOCH | 50 |  number of epochs required typically depends on size of N_SAMPLES (larger requires fewer epochs per stage) |
| MAGNITUDES | 5, 15, 25 | ImgAug magnitudes, adjusted progressively |
| DROPOUTS | 0.10, 0.20, 0.30 | Dropout rates, adjusted progressively |
| SHAPES | 224, 224, 224 | Image sizes: MN-V3-S = 224, EN-B0 = 224, EN-V2S = 300, EN-V2M = 384, EN-V2L = 480 |
| BATCH_SIZES | 4, 4, 4 | Mini-batch sizes (depends on GPU memory) |

