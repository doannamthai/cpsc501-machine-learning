from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Get dataset
def get_dataset(file_path, label, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=1, 
      label_name=label,
      #na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
    return dataset


class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features
    return features, labels

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

def main():
    LABEL_COLUMN = 'chd'
    NUMERIC_FEATURES =  ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
    CATEGORIES = {
        'famhist': ['Present', 'Absent'],
    }
    # Get data
    raw_train_data = get_dataset('heart_train.csv', 
                            label=LABEL_COLUMN)
    raw_test_data = get_dataset('heart_test.csv', 
                            label=LABEL_COLUMN)
    packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
    packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))

    # Normalize
    desc = pd.read_csv('heart.csv')[NUMERIC_FEATURES].describe()
    normalizer = functools.partial(normalize_numeric_data, mean=np.array(desc.T['mean']), std=np.array(desc.T['std']))

    # Grouping numeric data
    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]

    # Categorize 
    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    # Input layer
    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

    #Create model
    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(512, activation='elu',
                     kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.fit(packed_train_data, epochs=20)
    test_loss, test_accuracy = model.evaluate(packed_test_data)    
    print(f"Model Loss:    {test_loss:.2f}")
    print(f"Model Accuray: {test_accuracy*100:.1f}%")


if __name__ == "__main__":
    main()
