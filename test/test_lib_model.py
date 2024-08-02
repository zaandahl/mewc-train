import unittest
from unittest.mock import patch, MagicMock
from lib_model import (
    configure_optimizer_and_loss,
    build_classifier,
    unfreeze_model,
    fit_frozen,
    fit_progressive,
    calc_class_metrics,
    find_unfreeze_points
)
from lib_data import create_tensorset
import numpy as np
import os

class TestLibModel(unittest.TestCase):

    @patch('lib_model.optimizers')
    @patch('lib_model.losses')
    def test_configure_optimizer_and_loss(self, mock_losses, mock_optimizers):
        config = {
            'LR_SCHEDULER': 'exponential',
            'OPTIMIZER': 'adamw',
            'OPTIM_REG': 0.01,
            'LEARNING_RATE': 0.001,
            'PROG_STAGE_LEN': 10,
            'BATCH_SIZE': 32,
            'PROG_TOT_EPOCH': 50
        }
        num_classes = 3
        df_size = 1000
        optimizer, loss_f, act_f = configure_optimizer_and_loss(config, num_classes, df_size)

        self.assertIsNotNone(optimizer)
        self.assertIsNotNone(loss_f)
        self.assertEqual(act_f, 'softmax')

    @patch('lib_model.models')
    @patch('lib_model.hf_hub_download')
    def test_build_classifier(self, mock_hf_hub_download, mock_models):
        config = {
            'MODEL': 'enb0',
            'REPO_ID': 'repo_id',
            'FILENAME': 'filename',
            'LR_SCHEDULER': 'exponential',
            'OPTIMIZER': 'adamw',
            'OPTIM_REG': 0.01,
            'LEARNING_RATE': 0.001,
            'PROG_STAGE_LEN': 10,
            'BATCH_SIZE': 32,
            'PROG_TOT_EPOCH': 50,
            'DROPOUTS': [0.2]
        }
        num_classes = 3
        df_size = 1000
        img_size = 224

        mock_hf_hub_download.return_value = 'mock_filepath'
        mock_base_model = MagicMock()
        mock_models.load_model.return_value = MagicMock(layers=[mock_base_model])
        model = build_classifier(config, num_classes, df_size, img_size)

        self.assertIsNotNone(model)
        self.assertTrue(mock_models.load_model.called)

    @patch('lib_model.models')
    def test_unfreeze_model(self, mock_models):
        config = {
            'MODEL': 'enb0',
            'BUF': 1,
            'OPTIMIZER': 'adamw',
            'OPTIM_REG': 0.01,
            'LEARNING_RATE': 0.001,
            'PROG_STAGE_LEN': 10,
            'BATCH_SIZE': 32,
            'PROG_TOT_EPOCH': 50,
            'LR_SCHEDULER': 'exponential'
        }
        num_classes = 3
        df_size = 1000
        model = MagicMock()
        model.layers[0].layers = [MagicMock(name='block1_0_conv_pw_conv2d'), MagicMock(name='block2_0_conv_pw_conv2d')]

        with patch('lib_model.find_unfreeze_points', return_value=['block2_0_conv_pw_conv2d']):
            unfreeze_model(config, model, num_classes, df_size)
            self.assertTrue(model.layers[0].trainable)

    @patch('lib_model.create_tensorset')
    @patch('lib_model.callbacks')
    @patch('lib_model.unfreeze_model')
    def test_fit_frozen(self, mock_unfreeze_model, mock_callbacks, mock_create_tensorset):
        config = {
            'BATCH_SIZE': 32,
            'FROZ_EPOCH': 5,
            'LR_SCHEDULER': 'exponential',
            'OPTIMIZER': 'adamw',
            'OPTIM_REG': 0.01,
            'LEARNING_RATE': 0.001,
            'PROG_STAGE_LEN': 10,
            'PROG_TOT_EPOCH': 50
        }
        model = MagicMock()
        train_df = MagicMock()
        val_df = MagicMock()
        num_classes = 3
        df_size = 1000
        img_size = 224

        mock_create_tensorset.return_value = MagicMock()
        hist, model = fit_frozen(config, model, train_df, val_df, num_classes, df_size, img_size)

        self.assertTrue(mock_unfreeze_model.called)
        self.assertTrue(mock_create_tensorset.called)
        self.assertIsNotNone(hist)
        self.assertIsNotNone(model)

    @patch('lib_model.create_tensorset')
    @patch('lib_model.saving.save_model')
    def test_fit_progressive(self, mock_save_model, mock_create_tensorset):
        config = {
            'PROG_STAGE_LEN': 5,
            'PROG_TOT_EPOCH': 10,
            'BATCH_SIZE': 32,
            'NUM_AUG': 2,
            'DROPOUTS': [0.2, 0.3],
            'MAGNITUDES': [0.1, 0.2],
            'SAVEFILE': 'test_model',
            'MODEL': 'enb0',
            'LR_SCHEDULER': 'exponential',
            'OPTIMIZER': 'adamw',
            'OPTIM_REG': 0.01,
            'LEARNING_RATE': 0.001
        }
        model = MagicMock()
        train_df = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        val_df = MagicMock()
        output_fpath = '/code/output'
        img_size = 224

        mock_create_tensorset.return_value = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {'val_loss': [0.5, 0.4, 0.3, 0.2, 0.1]}
        model.fit.return_value = mock_history

        hhs, model, best_model_fpath = fit_progressive(config, model, train_df, val_df, output_fpath, img_size)

        self.assertTrue(mock_save_model.called)
        self.assertTrue(mock_create_tensorset.called)
        self.assertIsNotNone(hhs)
        self.assertIsNotNone(model)
        self.assertIsNotNone(best_model_fpath)

    @patch('lib_model.models.load_model')
    @patch('lib_model.utils.image_dataset_from_directory')
    @patch('lib_model.metrics.classification_report')
    @patch('lib_model.metrics.ConfusionMatrixDisplay')
    @patch('os.makedirs')
    @patch('os.path.join', side_effect=lambda *args: '/code/output/confusion_matrix.png')
    def test_calc_class_metrics(self, mock_path_join, mock_makedirs, mock_ConfusionMatrixDisplay, mock_classification_report, mock_image_dataset_from_directory, mock_load_model):
        model_fpath = 'path/to/model'
        test_fpath = 'path/to/test'
        output_fpath = '/code/output'
        classes = ['class1', 'class2', 'class3']
        batch_size = 32
        img_size = 224

        mock_load_model.return_value = MagicMock()
        mock_image_dataset_from_directory.return_value = MagicMock()
        mock_classification_report.return_value = 'mock_classification_report'
        mock_conf_matrix = MagicMock()
        mock_ConfusionMatrixDisplay.return_value = mock_conf_matrix

        if not os.path.exists(output_fpath):
            os.makedirs(output_fpath)

        calc_class_metrics(model_fpath, test_fpath, output_fpath, classes, batch_size, img_size)

        self.assertTrue(mock_load_model.called)
        self.assertTrue(mock_image_dataset_from_directory.called)
        self.assertTrue(mock_classification_report.called)
        self.assertTrue(mock_ConfusionMatrixDisplay.called)
        mock_makedirs.assert_called_once_with('/code/output', exist_ok=True)
        mock_path_join.assert_called()

if __name__ == '__main__':
    unittest.main()

