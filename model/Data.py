import os
import tensorflow as tf
from glob import glob
from preprocessing import load_audio , load_image


Audiofiles=r'C:\Users\hp\Desktop\coding\ML projects\Audio_Classifications\Multimodular\Cat&Dog\audio'
Imagefiles=r'C:\Users\hp\Desktop\coding\ML projects\Audio_Classifications\Multimodular\Cat&Dog\image'

def Audio_classification_data(batch_size=16):
        label_map = {
        'cat': 0,
        'dog': 1,
                   }


        def train_labeled_dataset(folder_name, label):
            path_pattern = os.path.join(Audiofiles,'training' ,folder_name, '*.wav')  
            print(f"Looking for files in: {path_pattern}")  # Optional debug
            files = tf.data.Dataset.list_files(path_pattern, shuffle=False)
            file_count = tf.data.experimental.cardinality(files).numpy()
            labels = tf.data.Dataset.from_tensor_slices(tf.repeat(label, file_count))
            return tf.data.Dataset.zip((files, labels))

        def test_labeled_dataset(folder_name, label):
            path_pattern = os.path.join(Audiofiles,'validation' ,folder_name, '*.wav')  
            print(f"Looking for files in: {path_pattern}")  # Optional debug
            files = tf.data.Dataset.list_files(path_pattern, shuffle=False)
            file_count = tf.data.experimental.cardinality(files).numpy()
            labels = tf.data.Dataset.from_tensor_slices(tf.repeat(label, file_count))
            return tf.data.Dataset.zip((files, labels))
        
        
        dataset_test = None
        for name, label in label_map.items():
            ds = test_labeled_dataset(name, label)
            if dataset_test is None:
                dataset_test = ds
            else:
                dataset_test = dataset_test.concatenate(ds)
                
        dataset_train = None
        for name, label in label_map.items():
            ds = train_labeled_dataset(name, label)
            if dataset_train is None:
                dataset_train = ds
            else:
                dataset_train = dataset_train.concatenate(ds)
        dataset_train = dataset_train.map(load_audio).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        dataset_test = dataset_test.map(load_audio).shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset_train , dataset_test

def image_classification_data(batch_size=32):
        label_map = {
            'cats': 0,
            'dogs': 1,
                    }


        def train_labeled_dataset(folder_name, label):
            path_pattern = os.path.join(Imagefiles,'training_set' ,folder_name, '*.jpg')  
            print(f"Looking for files in: {path_pattern}")  # Optional debug
            files = tf.data.Dataset.list_files(path_pattern, shuffle=False)
            file_count = tf.data.experimental.cardinality(files).numpy()
            labels = tf.data.Dataset.from_tensor_slices(tf.repeat(label, file_count))
            return tf.data.Dataset.zip((files, labels))

        def test_labeled_dataset(folder_name, label):
            path_pattern = os.path.join(Imagefiles,'test_set' ,folder_name, '*.jpg')  
            print(f"Looking for files in: {path_pattern}")  # Optional debug
            files = tf.data.Dataset.list_files(path_pattern, shuffle=False)
            file_count = tf.data.experimental.cardinality(files).numpy()
            labels = tf.data.Dataset.from_tensor_slices(tf.repeat(label, file_count))
            return tf.data.Dataset.zip((files, labels))
        
        
        dataset_test = None
        for name, label in label_map.items():
            ds = test_labeled_dataset(name, label)
            if dataset_test is None:
                dataset_test = ds
            else:
                dataset_test = dataset_test.concatenate(ds)

    
        dataset_train = None
        for name, label in label_map.items():
            ds = train_labeled_dataset(name, label)
            if dataset_train is None:
                dataset_train = ds
            else:
                dataset_train = dataset_train.concatenate(ds)
        dataset_train = dataset_train.map(load_image).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        dataset_test = dataset_test.map(load_image).shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset_train , dataset_test
    
   