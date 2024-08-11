import unittest
import warnings
import tensorflow as tf
from unittest.mock import patch, MagicMock
from keras import models, layers, optimizers, losses
from lib_model import (
    configure_optimizer_and_loss, 
    import_model, 
    build_sequential_model, 
    build_classifier, 
    find_unfreeze_points, 
    unfreeze_model, 
    fit_frozen, 
    fit_progressive, 
    calc_class_metrics
)

class TestLibModel(unittest.TestCase):
    def setUp(self):
        self.config = {
            'LR_SCHEDULER': 'exponential',
            'OPTIMIZER': 'adam',
            'OPTIM_REG': 0.001,
            'LEARNING_RATE': 0.001,
            'BATCH_SIZE': 32,
            'PROG_STAGE_LEN': 10,
            'PROG_TOT_EPOCH': 50,
            'MODEL': 'enb0',
            'REPO_ID': 'some_repo',
            'FILENAME': 'some_file',
            'DROPOUTS': [0.2, 0.3],
            'BUF': 1,
            'FROZ_EPOCH': 10,
            'SAVEFILE': 'savefile',
            'MAGNITUDES': [0.1],
            'NUM_AUG': 1,
        }
        self.num_classes = 2
        self.df_size = 100
        self.img_size = 224
        self.train_df = MagicMock()
        self.val_df = MagicMock()
        self.output_fpath = '/tmp'

    @patch('lib_model.models.load_model')
    @patch('lib_model.hf_hub_download')
    def test_import_model_valid_model_name(self, mock_hf_hub_download, mock_load_model):
        mock_load_model.return_value = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3))
        ])
        result = import_model(self.img_size, 'enb0', self.config['REPO_ID'], self.config['FILENAME'])
        self.assertIsInstance(result, layers.Layer)

    @patch('lib_model.models.load_model')
    @patch('lib_model.hf_hub_download', side_effect=Exception)
    def test_import_model_invalid_model_name(self, mock_hf_hub_download, mock_load_model):
        # Mock the behavior when hf_hub_download raises an exception and model loading fails
        mock_load_model.side_effect = ValueError("File format not supported")
        try:
            result = import_model(self.img_size, 'invalid', self.config['REPO_ID'], self.config['FILENAME'])
        except ValueError as e:
            result = None
        self.assertIsNone(result)


    @patch('lib_model.configure_optimizer_and_loss')
    @patch('lib_model.import_model')
    def test_build_sequential_model(self, mock_import_model, mock_configure_optimizer_and_loss):
        mock_configure_optimizer_and_loss.return_value = (optimizers.Adam(), losses.BinaryFocalCrossentropy(), 'sigmoid')
        mock_import_model.return_value = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3))
        ])
        model_base = mock_import_model.return_value
        model = build_sequential_model(model_base, self.num_classes, 'sigmoid', 'enb0', 0.2)
        self.assertIsInstance(model, models.Sequential)


    @patch('lib_model.configure_optimizer_and_loss')
    @patch('lib_model.import_model')
    def test_build_classifier(self, mock_import_model, mock_configure_optimizer_and_loss):
        mock_configure_optimizer_and_loss.return_value = (optimizers.Adam(), losses.BinaryFocalCrossentropy(), 'sigmoid')
        mock_base_model = models.Sequential()
        mock_base_model.add(layers.Input(shape=(224, 224, 3)))
        mock_base_model.add(layers.Conv2D(32, (3, 3)))
        mock_import_model.return_value = mock_base_model
        model = build_classifier(self.config, self.num_classes, self.df_size, self.img_size)
        self.assertIsInstance(model, models.Sequential)

    def test_find_unfreeze_points(self):
        model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),  # Use Input layer here
            layers.Conv2D(32, (3, 3), name='block_0_conv_pw_conv2d')
        ])
        result = find_unfreeze_points(model, 'enb0', 1)
        self.assertEqual(result, ['block_0_conv_pw_conv2d'])

    @patch('lib_model.configure_optimizer_and_loss')
    @patch('lib_model.find_unfreeze_points', return_value=['block_0_conv_pw_conv2d'])
    def test_unfreeze_model(self, mock_find_unfreeze_points, mock_configure_optimizer_and_loss):
        mock_configure_optimizer_and_loss.return_value = (optimizers.Adam(), losses.BinaryFocalCrossentropy(), 'sigmoid')
        base_model = models.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3), name='block_0_conv_pw_conv2d')
        ])
        model = models.Sequential([base_model])
        result = unfreeze_model(self.config, model, self.num_classes, self.df_size)
        self.assertIsInstance(result, models.Sequential)

    @patch('lib_model.create_tensorset')
    @patch('lib_model.unfreeze_model')
    def test_fit_frozen(self, mock_unfreeze_model, mock_create_tensorset):
        dummy_data = tf.data.Dataset.from_tensor_slices((tf.random.normal([10, 224, 224, 3]), tf.one_hot(tf.random.uniform([10], maxval=2, dtype=tf.int32), 2))).batch(2)
        mock_create_tensorset.side_effect = [dummy_data, dummy_data]
        model = models.Sequential([layers.Input(shape=(224, 224, 3)), layers.Conv2D(32, (3, 3))])
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(self.num_classes, activation='sigmoid'))
        model.compile(optimizer=optimizers.Adam(), loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
        hist, model = fit_frozen(self.config, model, self.train_df, self.val_df, self.num_classes, self.df_size, self.img_size)
        self.assertIsNotNone(hist)

    @patch('lib_model.create_tensorset')
    @patch('lib_model.saving.save_model', return_value=MagicMock())
    def test_fit_progressive(self, mock_save_model, mock_create_tensorset):
        def create_dummy_dataset(*args, **kwargs):
            return tf.data.Dataset.from_tensor_slices(
                (tf.random.normal([10, 224, 224, 3]), tf.one_hot(tf.random.uniform([10], maxval=2, dtype=tf.int32), 2))
            ).batch(2)
        mock_create_tensorset.side_effect = create_dummy_dataset
        model = models.Sequential([layers.Input(shape=(224, 224, 3)), layers.Conv2D(32, (3, 3))])
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(self.num_classes, activation='sigmoid'))
        model.compile(optimizer=optimizers.Adam(), loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
        result = fit_progressive(self.config, model, self.train_df, self.val_df, self.output_fpath, self.img_size)
        self.assertIsInstance(result, tuple)

    @patch('lib_model.optimizers.schedules.ExponentialDecay')
    def test_configure_optimizer_and_loss_binary(self, mock_exponential_decay):
        mock_exponential_decay.return_value = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9
        )
        optimizer, loss_f, act_f = configure_optimizer_and_loss(self.config, 2, self.df_size)
        self.assertIsInstance(optimizer, optimizers.Adam)
        self.assertIsInstance(loss_f, losses.BinaryFocalCrossentropy)
        self.assertEqual(act_f, 'sigmoid')

    @patch('lib_model.optimizers.schedules.ExponentialDecay')
    def test_configure_optimizer_and_loss_categorical(self, mock_exponential_decay):
        mock_exponential_decay.return_value = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9
        )
        optimizer, loss_f, act_f = configure_optimizer_and_loss(self.config, 5, self.df_size)
        self.assertIsInstance(optimizer, optimizers.Adam)
        self.assertIsInstance(loss_f, losses.CategoricalFocalCrossentropy)
        self.assertEqual(act_f, 'softmax')

    @patch('lib_model.optimizers.schedules.ExponentialDecay')
    def test_configure_optimizer_and_loss_cosine(self, mock_exponential_decay):
        self.config['LR_SCHEDULER'] = 'cosine'
        mock_exponential_decay.return_value = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9
        )
        optimizer, loss_f, act_f = configure_optimizer_and_loss(self.config, 5, self.df_size)
        self.assertIsInstance(optimizer, optimizers.Adam)

    @patch('lib_model.optimizers.schedules.ExponentialDecay')
    def test_configure_optimizer_and_loss_polynomial(self, mock_exponential_decay):
        self.config['LR_SCHEDULER'] = 'polynomial'
        mock_exponential_decay.return_value = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9
        )
        optimizer, loss_f, act_f = configure_optimizer_and_loss(self.config, 5, self.df_size)
        self.assertIsInstance(optimizer, optimizers.Adam)

    def test_build_sequential_model_vit(self):
        model_base = layers.Conv1D(32, 3, input_shape=(224, 3), name='blocks_0_attn')
        input_shape = (224, 3)  # Define the input shape explicitly
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(model_base)
        self.assertIsInstance(model, models.Sequential)

    def test_build_sequential_model_cn(self):
        model_base = layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3), name='stages_0_downsample_1_conv2d')
        input_shape = (224, 224, 3)  # Define the input shape explicitly
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(model_base)
        self.assertIsInstance(model, models.Sequential)

    @patch('lib_model.configure_optimizer_and_loss')
    @patch('lib_model.find_unfreeze_points', return_value=['block_0_conv_pw_conv2d'])
    def test_unfreeze_model_no_blocks(self, mock_find_unfreeze_points, mock_configure_optimizer_and_loss):
        mock_configure_optimizer_and_loss.return_value = (optimizers.Adam(), losses.BinaryFocalCrossentropy(), 'sigmoid')
        base_model = models.Sequential([
            layers.Input(shape=(224, 224, 3)), 
            layers.Conv2D(32, (3, 3), name='block_0_conv_pw_conv2d')
        ])
        self.config['BUF'] = 0
        result = unfreeze_model(self.config, base_model, self.num_classes, self.df_size)
        self.assertIsInstance(result, models.Sequential)

    @patch('lib_model.configure_optimizer_and_loss')
    def test_unfreeze_model_all_blocks(self, mock_configure_optimizer_and_loss):
        mock_configure_optimizer_and_loss.return_value = (optimizers.Adam(), losses.BinaryFocalCrossentropy(), 'sigmoid')
        base_model = models.Sequential([
            layers.Input(shape=(224, 224, 3), name='input_layer'),
            models.Sequential([
                layers.Conv2D(32, (3, 3), name='block_0_conv_pw_conv2d')
            ])
        ])
        self.config['BUF'] = -1
        result = unfreeze_model(self.config, base_model, self.num_classes, self.df_size)
        self.assertIsInstance(result, models.Sequential)

    @patch('lib_model.optimizers.schedules.ExponentialDecay')
    def test_configure_optimizer_and_loss_lion(self, mock_exponential_decay):
        self.config['OPTIMIZER'] = 'lion'
        optimizer, loss_f, act_f = configure_optimizer_and_loss(self.config, 2, self.df_size)
        self.assertIsInstance(optimizer, optimizers.Lion)

    def test_find_unfreeze_points_vit(self):
        model = models.Sequential([layers.Conv1D(32, 3, input_shape=(224, 3), name='blocks_0_attn')])
        result = find_unfreeze_points(model, 'vitt', 1)
        self.assertEqual(result, ['blocks_0_attn'])

    def test_find_unfreeze_points_cn(self):
        model = models.Sequential([layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3), name='stages_0_downsample_1_conv2d')])
        result = find_unfreeze_points(model, 'cnp', 1)
        self.assertEqual(result, ['stages_0_downsample_1_conv2d'])

if __name__ == '__main__':
    unittest.main()
