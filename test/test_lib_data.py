import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
from pathlib import Path
from lib_data import (ensure_output_directory, process_samples_from_config, process_custom_sample_file, check_upload_format, validate_directory_structure, 
                      print_dsinfo, create_dataframe, create_train, create_fixed, create_tensorset, load_img)

class TestLibData(unittest.TestCase):

    def test_ensure_output_directory(self):
        test_path = "test_dir"
        if os.path.exists(test_path):
            os.rmdir(test_path)
        ensure_output_directory(test_path)
        self.assertTrue(os.path.exists(test_path))
        os.rmdir(test_path)

    def test_process_samples_from_config(self):
        config = {
            'CLASS_SAMPLES_DEFAULT': 10,
            'CLASS_SAMPLES_SPECIFIC': [
                {'SAMPLES': 5, 'CLASSES': ['class1', 'class2']},
                {'SAMPLES': 3, 'CLASSES': ['class3']}
            ]
        }
        expected_result = {
            'default': 10,
            'class1': 5,
            'class2': 5,
            'class3': 3
        }
        result, is_custom_sample = process_samples_from_config(config)
        self.assertEqual(result, expected_result)
        self.assertTrue(is_custom_sample)

    def test_process_custom_sample_file(self):
        custom_sample_file = {
            'default': 10,
            'specific': {
                5: ['class1', 'class2'],
                3: ['class3']
            }
        }
        expected_result = {
            'default': 10,
            'class1': 5,
            'class2': 5,
            'class3': 3
        }
        result, is_custom_sample = process_custom_sample_file(custom_sample_file)
        self.assertEqual(result, expected_result)
        self.assertTrue(is_custom_sample)

    def test_check_upload_format(self):
        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch('os.listdir', return_value=['subdir']):
                    with patch('os.path.isdir', return_value=True):
                        with patch('os.listdir', return_value=['image.jpg']):
                            self.assertTrue(check_upload_format('main_directory'))
        
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                check_upload_format('main_directory')

        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=False):
                with self.assertRaises(NotADirectoryError):
                    check_upload_format('main_directory')

        with patch('os.path.exists', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch('os.listdir', return_value=['subdir']):
                    with patch('os.path.isdir', return_value=True):
                        with patch('os.listdir', return_value=['file.txt']):
                            with self.assertRaises(ValueError):
                                check_upload_format('main_directory')

    def test_validate_directory_structure(self):
        with patch('lib_data.check_upload_format', return_value=True):
            with patch('sys.exit') as exit_mock:
                validate_directory_structure('train_path', 'val_path', 'test_path')
                exit_mock.assert_not_called()

        with patch('lib_data.check_upload_format', side_effect=FileNotFoundError):
            with patch('sys.exit') as exit_mock:
                validate_directory_structure('train_path', 'val_path', 'test_path')
                exit_mock.assert_called_once()

    def test_print_dsinfo(self):
        data = {'File': ['file1', 'file2'], 'Label': ['class1', 'class2']}
        df = pd.DataFrame(data)
        with patch('builtins.print') as mocked_print:
            print_dsinfo(df, ds_name='test_ds')
            mocked_print.assert_any_call('Dataset: test_ds')
            mocked_print.assert_any_call(f'Number of images in the dataset: {df.shape[0]}')

    def test_create_dataframe(self):
        with patch('lib_data.Path.glob', return_value=['path/class1/file1.jpg', 'path/class2/file2.jpg']):
            df = create_dataframe('dataset_path', 1, 12345, False, None)
            self.assertEqual(df.shape[0], 2)
            self.assertIn('File', df.columns)
            self.assertIn('Label', df.columns)

    def test_create_train(self):
        with patch('lib_data.create_dataframe', return_value=pd.DataFrame({'File': ['file1', 'file2'], 'Label': ['class1', 'class2']})):
            train_df, num_classes = create_train('dataset_path')
            self.assertEqual(num_classes, 2)
            self.assertEqual(train_df.shape[0], 2)
            self.assertIn('File', train_df.columns)
            self.assertIn('Label', train_df.columns)

    def test_create_fixed(self):
        with patch('lib_data.Path.glob', return_value=['path/class1/file1.jpg', 'path/class2/file2.jpg']):
            df = create_fixed('dataset_path')
            self.assertEqual(df.shape[0], 2)
            self.assertIn('File', df.columns)
            self.assertIn('Label', df.columns)

    def test_create_tensorset(self):
        data = {'File': ['path/class1/file1.jpg', 'path/class2/file2.jpg'], 'Label': ['class1', 'class2']}
        df = pd.DataFrame(data)
        dataset = create_tensorset(df, 224, 32, 0.1, 1, "train")
        self.assertIsInstance(dataset, tf.data.Dataset)

    def test_load_img(self):
        test_image_path = 'path/to/test_image.png'
        test_img_size = 224
        
        # Create a mock tensor for the image
        mock_tensor = tf.random.uniform((test_img_size, test_img_size, 3), dtype=tf.float32)
        
        with patch('tensorflow.io.read_file', return_value=MagicMock()), \
             patch('tensorflow.image.decode_png', return_value=mock_tensor), \
             patch('tensorflow.image.convert_image_dtype', return_value=mock_tensor), \
             patch('tensorflow.image.resize', return_value=mock_tensor):
            img = load_img(test_image_path, test_img_size)
            self.assertIsInstance(img, tf.Tensor)

if __name__ == '__main__':
    unittest.main()

