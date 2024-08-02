import unittest
import os
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
        mock_listdir.return_value = ['class1', 'class2']
        
        with patch('os.listdir', side_effect=[['image1.jpg', 'image2.png'], ['image3.jpg', 'image4.png']]):
            validate_directory_structure('train_path', 'val_path', 'test_path')

    def test_print_dsinfo(self):
        ds_df = pd.DataFrame({
            'File': ['file1.jpg', 'file2.jpg', 'file3.jpg'],
            'Label': ['class1', 'class1', 'class2']
        })
        with patch('builtins.print') as mocked_print:
            print_dsinfo(ds_df, 'test_dataset')
            mocked_print.assert_any_call('Dataset: test_dataset')
            mocked_print.assert_any_call('Number of images in the dataset: 3')
            mocked_print.assert_any_call("class1    2\nclass2    1\nName: Label, dtype: int64\n")

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