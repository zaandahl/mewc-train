<img src="mewc_logo_hex.png" alt="MEWC Hex Sticker" width="200" align="right"/>

# mewc-train

## Introduction
This repository contains code to build a Docker container for running mewc-train. This is a tool used to train a model for predicting species from camera trap images. The classifier engine used in mewc-train is EfficientNetV2. 

You can supply arguments via an environment file where the contents of that file are in the following format with one entry per line:
```
VARIABLE=VALUE
```

## Version 2 Updates

The `mewc-train` Docker image has been updated to version 2. Key updates include:

- **Base Image**: Uses the new `mewc-flow` base image featuring `tensorflow/tensorflow:2.16.1-gpu`, `CUDA`, `cuDNN`, and `JAX`.
- **Hugging Face Base Models**: Start training off a base model from [Hugging Face](https://huggingface.co/models). 
- **Improved Training Control**: New user configurable options to fine tune model training.
- **Optional Validation Path**: There is an option to specify separate paths for data|validation `VAL_PATH` and data|test `TEST_PATH`. Doing so will keep test data isolated from hyper-parameter tuning leakage.

For users who wish to continue using version 1, the older Dockerfile and requirements can still be accessed by checking out the `v1.0.10` tag:

```bash
git checkout v1.0.10
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

1. Trained Model (`$SAVEFILE_$MODEL_best.keras`): This is the serialized version of your trained neural network and contains all the learned weights and biases. This file is crucial for making predictions on new, unseen data.

2. Class List (`$SAVEFILE_class_map.yaml`): A YAML file that provides a mapping between class names and class IDs. This is especially vital to ensure that predictions made by the model are correctly associated with their respective class names.

3. Confusion Matrix (`confusion_matrix.png`): A confusion matrix is a table used in machine learning to evaluate the performance of a classification model. The confusion matrix helps in understanding how well the model is classifying instances into different classes. By examining the confusion matrix, you can identify which classes are being misclassified and understand the patterns of errors. This file is not required for making predictions.

When using `mewc-predict`, it expects both `$SAVEFILE_$MODEL_best.keras` and `$SAVEFILE_class_map.yaml` to be available. This ensures seamless predictions and accurate class labeling. Ensure you maintain the integrity of these files and store them securely to make the most out of your trained model.

## Config Options

The following environment variables are supported for configuration, with their default values shown. You can omit any variables you don't need to change. If you want to use all default values, you can leave the `--env-file $ENV_FILE` option out of the Docker command altogether.

The main volume mount in the Docker command maps your local data directory to the `/data` directory in the container. The default values below assume you have a directory structure as shown above. Remember that all paths are relative inside the Docker container, so `/data` exists within the container and is not a local path on your machine.

Output from the container will be saved in the `/data/output/$SAVEFILE/$MODEL/` directory. The class list derived from the train/test directory structure will be saved to `$SAVEFILE_class_map.yaml`. The final output model file will be saved as `$SAVEFILE_$MODEL_final.keras`. During training, the best-performing model after each progressive stage is saved as `$SAVEFILE_$MODEL_best.keras`. The best-performing model should be the model selected for inference as the final model may be overfit.

Note that for the `MAGNITUDES`, `DROPOUTS`, and other list-type variables, you can supply multiple values separated by commas. These values will be used in sequence for each progressive training stage. For example, if you supply `MAGNITUDES=0.2,0.4,0.6,0.8`, then the first stage will use a magnitude of 0.2, the second stage will use 0.4, and so on. The length of these variables must match. For example, in the default values shown below, there are four values for each variable.

### Supported Environment Variables

| Variable               | Default                           | Description |
| ---------------------- | --------------------------------- | ----------- |
| SEED                   | 12345                             | Random seed for reproducibility of sampled datasets and model initialization |
| MODEL                  | 'ENB0'                             | Model architecture: EN:[B0,B2,S,M,L,XL], CN:[P,N,T,S,B,L], ViT:[T,S,B,L], or a pretrained filename |
| SAVEFILE               | 'case_study'                      | Filename to save the `.keras` model; MODEL name is appended automatically |
| OUTPUT_PATH            | '/data/output'                    | Path to save output files (model, class map, confusion matrix) |
| REPO_ID                | 'bwbrook/mewc_pretrained'         | Hugging Face model repository ID for fine-tuning on training data |
| FILENAME               | 'NA'                              | Hugging Face filename for the base model, or "NA" for no pretrained model |
| TRAIN_PATH             | '/data/train'                     | Path to training data (subfolders for each class) |
| TEST_PATH              | '/data/test'                      | Path to hold-out test data (for final model evaluation) |
| VAL_PATH               | '/data/test'                      | Path to validation data (optional, for hyperparameter tuning) |
| OPTIMIZER              | 'adamw'                           | Optimizer algorithm: 'adamw', 'rmsprop', or 'lion' |
| OPTIM_REG              | 1e-4                              | Regularization (weight decay) parameter for the optimizer |
| LR_SCHEDULER           | 'expon'                           | Learning-rate scheduler: 'expon', 'cosine', 'polynomial' |
| LEARNING_RATE          | 1e-4                              | Initial learning rate (default: 1e-4) |
| BATCH_SIZE             | 16                                | Mini-batch size (adjust based on GPU memory) |
| NUM_AUG                | 3                                 | Number of per-image random augmentation layers (default: 3, suggested range: 1-5) |
| FROZ_EPOCH             | 10                                | Number of epochs to train the frozen model before fine-tuning |
| BUF                    | 2                                 | Blocks to unfreeze for fine-tuning (suggest 1-2 for EN/CN, 5-9 for ViT) |
| PROG_STAGE_LEN         | 10                                | Number of progressive fine-tuning epochs prior to the final stage |
| PROG_TOT_EPOCH         | 60                                | Total number of epochs (depends on size of class_samples.yaml) |
| MAGNITUDES             | 0.2, 0.4, 0.6, 0.8                | RandAug magnitudes, increased progressively (range 0-1) |
| DROPOUTS               | 0.1, 0.2, 0.3, 0.4                | Dropout rates, increased progressively (range 0-1) |
| CLASS_SAMPLES_DEFAULT  | 4000                              | Default number of sample images per class to be trained each epoch |
| CLASS_SAMPLES_SPECIFIC | *None* (commented out by default) | Specific number of samples for each class (see example below) |

### CLASS_SAMPLES_SPECIFIC Example

To specify different sample sizes for specific classes, use the following environment variable format:

```plaintext
CLASS_SAMPLES_SPECIFIC='[ \
    {"SAMPLES": 3000, "CLASS": "dog"}, \
    {"SAMPLES": 3000, "CLASS": "horse"}, \
    {"SAMPLES": 2000, "CLASS": "cow"}, \
    {"SAMPLES": 1000, "CLASS": "pademelon"}, \
    {"SAMPLES": 1000, "CLASS": "walrus"}, \
    {"SAMPLES": 1000, "CLASS": "gnat"} \
]'
```

Simply include this varaible in your `$ENV_FILE` file, customise it to your models class names and desired sample sizes and pass it to Docker as usual:

```plaintext
docker run --env-file `$ENV_FILE` --volume /mnt/mewc-volume/train/data:/data zaandahl/mewc-train
```

## GitHub Actions and DockerHub
This project uses GitHub Actions to automate the build process and push the Docker image to DockerHub. You can find the image at:

- [zaandahl/mewc-train DockerHub Repository](https://hub.docker.com/repository/docker/zaandahl/mewc-train)

For users needing the older version, the v1.0.10 image is also available on DockerHub by using the appropriate tag:

```bash
docker pull zaandahl/mewc-train:v1.0.10
```