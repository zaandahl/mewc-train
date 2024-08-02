import unittest
import os
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress other specific warnings
warnings.filterwarnings("ignore", message="Unable to register cuFFT factory")
warnings.filterwarnings("ignore", message="Unable to register cuDNN factory")
warnings.filterwarnings("ignore", message="Unable to register cuBLAS factory")

import pandas as pd
import tensorflow as tf
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from lib_data import (
    ensure_output_directory,
    process_custom_sample_file,
    validate_directory_structure,
    print_dsinfo,
    create_train,
    create_fixed,
    create_tensorset
)

class TestLibData(unittest.TestCase):

    def test_ensure_output_directory(self):
        test_dir = "test_dir"
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
        ensure_output_directory(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        os.rmdir(test_dir)

    def test_process_custom_sample_file(self):
        custom_sample_file = {
            'default': 10,
            'specific': {
                5: ['class1', 'class2'],
                3: ['class3']
            }
        }
        expected_output = {
            'default': 10,
            'class1': 5,
            'class2': 5,
            'class3': 3
        }
        result, is_custom_sample = process_custom_sample_file(custom_sample_file)
        self.assertEqual(result, expected_output)
        self.assertTrue(is_custom_sample)

    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_validate_directory_structure(self, mock_listdir, mock_isdir, mock_exists):
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.side_effect = [
            ['class1', 'class2'],  # First call for train_path
            ['image1.jpg', 'image2.png'],  # Second call for class1 in train_path
            ['image3.jpg', 'image4.png'],  # Third call for class2 in train_path
            ['class1', 'class2'],  # Fourth call for val_path
            ['image1.jpg', 'image2.png'],  # Fifth call for class1 in val_path
            ['image3.jpg', 'image4.png'],  # Sixth call for class2 in val_path
            ['class1', 'class2'],  # Seventh call for test_path
            ['image1.jpg', 'image2.png'],  # Eighth call for class1 in test_path
            ['image3.jpg', 'image4.png']  # Ninth call for class2 in test_path
        ]

        validate_directory_structure('train_path', 'val_path', 'test_path')

    def test_print_dsinfo(self):
        ds_df = pd.DataFrame({
            'File': ['file1.jpg', 'file2.jpg', 'file3.jpg'],
            'Label': ['class1', 'class1', 'class2']
        })
        with patch('builtins.print') as mocked_print:
            print_dsinfo(ds_df, 'test_dataset')
            actual_calls = [call.args[0] for call in mocked_print.call_args_list]
            expected_calls = [
                'Dataset: test_dataset',
                'Number of images in the dataset: 3',
                'Label\nclass1    2\nclass2    1\nName: count, dtype: int64\n'
            ]
            for expected, actual in zip(expected_calls, actual_calls):
                print(f"Expected: {expected}")
                print(f"Actual: {actual}")
            self.assertEqual(expected_calls, actual_calls)

    @patch('lib_data.Path.glob')
    def test_create_train(self, mock_glob):
        mock_glob.return_value = [Path(f'path/class{i}/image{j}.jpg') for i in range(1, 4) for j in range(1, 6)]
        train_df, num_classes = create_train('ds_path', ns=3)
        self.assertEqual(train_df.shape[0], 9)
        self.assertEqual(num_classes, 3)

    @patch('lib_data.Path.glob')
    def test_create_fixed(self, mock_glob):
        mock_glob.return_value = [Path(f'path/class{i}/image{j}.jpg') for i in range(1, 4) for j in range(1, 6)]
        ds_df = create_fixed('ds_path')
        self.assertEqual(ds_df.shape[0], 15)
        self.assertEqual(ds_df.columns.tolist(), ['File', 'Label'])

    @patch('lib_data.Path.glob')
    def test_create_tensorset(self, mock_glob):
        mock_glob.return_value = [Path(f'path/class{i}/image{j}.jpg') for i in range(1, 4) for j in range(1, 6)]
        ds_df = create_fixed('ds_path')
        dataset = create_tensorset(ds_df, img_size=224, batch_size=2)
        self.assertIsInstance(dataset, tf.data.Dataset)

if __name__ == '__main__':
    unittest.main()

