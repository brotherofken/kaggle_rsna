#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from object_detection.utils import dataset_util

# In[]
flags = tf.app.flags
flags.DEFINE_string('input_tf_record', None, 'Path to input train TFRecord.')
flags.DEFINE_string('input_labeling_path', None, 'Path to labels.')
flags.DEFINE_string('input_detailed_info', None, 'Path to detailed info.')
flags.DEFINE_string('output_prefix', None, 'Output file prefix. Output file will be "{prefix}.fold_{N}.[train|valid].tfrecord".')
flags.DEFINE_integer('n_splits', 10, 'Splits number.')
flags.DEFINE_integer('seed', 42, 'Seed for better model tuning.')
flags.DEFINE_integer('fold', 0, 'Fold to take, zero based.')
FLAGS = flags.FLAGS

# In[] Main
def main(_):
    assert(FLAGS.fold >= 0 and FLAGS.fold < FLAGS.n_splits)

# In[]
    detailed_class_info = pd.read_csv(FLAGS.input_labeling_path)
    train_labels = pd.read_csv(FLAGS.input_detailed_info)
    labeling = pd.merge(left = detailed_class_info, right = train_labels, how = 'left', on = 'patientId')
    labeling = labeling.drop_duplicates()

# In[] Encode labels
    lencoder = LabelEncoder()
    lencoder.fit(labeling['class'])
    assert(len(lencoder.classes_) == 3)
    labeling['class_int'] = lencoder.transform(labeling['class'])

# In[] Split dataset
    skf = StratifiedKFold(n_splits=FLAGS.n_splits, shuffle=True, random_state=FLAGS.seed)
    labeling_for_split = labeling[['patientId','class_int']].drop_duplicates()
    train_index, valid_index = list(skf.split(labeling_for_split, labeling_for_split['class_int']))[FLAGS.fold]
    train_patients = set(labeling_for_split.iloc[train_index]['patientId'])
    valid_patients = set(labeling_for_split.iloc[valid_index]['patientId'])
    assert(len(train_patients & valid_patients) == 0)
    print('Unique patients in train fold: {}'.format(len(train_patients)))
    print('Unique patients in valid fold: {}'.format(len(valid_patients)))

# In[] Split train into two folds
    filename_train = '{}.fold_{}.train.tfrecord'.format(FLAGS.output_prefix, FLAGS.fold)
    filename_valid = '{}.fold_{}.valid.tfrecord'.format(FLAGS.output_prefix, FLAGS.fold)
    train_writer = tf.python_io.TFRecordWriter(filename_train)
    valid_writer = tf.python_io.TFRecordWriter(filename_valid)
    for record in tf.python_io.tf_record_iterator(FLAGS.input_tf_record):
        example = tf.train.Example()
        example.ParseFromString(record)
        filename = example.features.feature['image/source_id'].bytes_list.value[0].decode("utf-8")
        patient_id = os.path.basename(os.path.splitext(filename)[0])
        is_train = patient_id in train_patients
        is_valid = patient_id in valid_patients
        assert(is_train != is_valid)
        if is_train:
            train_writer.write(record)
        if is_valid:
            valid_writer.write(record)
    train_writer.close()
    valid_writer.close()

# In[]
if __name__ == '__main__':
    tf.app.run()
