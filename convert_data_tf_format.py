#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import itertools
import os
import queue
import threading

import numpy as np
import scipy as sp
import scipy.ndimage
import pydicom

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils.visualization_utils import encode_image_array_as_png_str, draw_bounding_boxes_on_image_array

# In[]
flags = tf.app.flags
flags.DEFINE_string('input_images_path', '', 'Path to input dcm images.')
flags.DEFINE_string('input_labeling_path', '', 'Path to labels.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord.')
flags.DEFINE_integer('threads', 1, 'Path to output TFRecord.')
flags.DEFINE_integer('take_first_n_elements', 0, 'Path to output TFRecord.')
FLAGS = flags.FLAGS

# Left here for debugging purposes
#    FLAGS.input_images_path = '/home/rakhunzy/workspace/projects/kaggle_rsna/rsna_data/stage_1_train_images'
#    FLAGS.input_labeling_path = '/home/rakhunzy/workspace/projects/kaggle_rsna/rsna_data/stage_1_train_labels.csv'
#    FLAGS.output_path = '/home/rakhunzy/workspace/projects/kaggle_rsna/rsna_data_preprocessed/stage_1_train_images.tfrecord'
#    FLAGS.input_images_path = '/home/rakhunzy/workspace/projects/kaggle_rsna/rsna_data/stage_1_test_images'
#    FLAGS.input_labeling_path = '/home/rakhunzy/workspace/projects/kaggle_rsna/rsna_data/stage_1_test_images.csv'
#    FLAGS.output_path = '/home/rakhunzy/workspace/projects/kaggle_rsna/rsna_data_preprocessed/stage_1_test_images.tfrecord'
#    FLAGS.threads = 6

# In[]
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def dicom_resample(dcm_data, new_spacing=[0.2, 0.2]):
    assert('SliceThickness' not in dcm_data)
    # Determine current pixel spacing, then estimate final image width and
    spacing = np.array(dcm_data.PixelSpacing)
    resize_factor = spacing / new_spacing
    new_real_shape = dcm_data.pixel_array.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / dcm_data.pixel_array.shape
    new_spacing = spacing / real_resize_factor
    image = sp.ndimage.interpolation.zoom(dcm_data.pixel_array,
                                          real_resize_factor)
    return image, new_spacing, real_resize_factor


def create_tf_example(dcm_path, bboxes):
    dcm_data = pydicom.read_file(dcm_path)
    image, new_spacing, resize_factor = dicom_resample(dcm_data)

    height = image.shape[0]
    width = image.shape[1]

    encoded_image_data = encode_image_array_as_png_str(image)
    image_format = b'png'

    xmins = [resize_factor[0] * bbox.x / width for bbox in bboxes]
    xmaxs = [resize_factor[0] * bbox.xmax() / width for bbox in bboxes]
    ymins = [resize_factor[1] * bbox.y / height for bbox in bboxes]
    ymaxs = [resize_factor[1] * bbox.ymax() / height for bbox in bboxes]

#    if bboxes:
#        np_bboxes = np.array([ymins, xmins, ymaxs, xmaxs]).T
#        draw_bounding_boxes_on_image_array(image, np_bboxes)
#        pylab.figure()
#        pylab.imshow(image)

    classes_text = [b'opacity' for bbox in bboxes]
    classes = [1 for bbox in bboxes]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(bytes(dcm_path, encoding='utf-8')),
      'image/source_id': dataset_util.bytes_feature(bytes(dcm_path, encoding='utf-8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/pixel_spacing': dataset_util.float_list_feature(new_spacing)
    }))
    return tf_example


class Bbox:
    def __init__(self, patientId, x=-1.0, y=-1.0, width=-1.0, height=-1.0,
                 Target=-1):
        self.patientId = patientId
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.Target = int(Target)

    def xmax(self):
        return self.x + self.width

    def ymax(self):
        return self.x + self.height


class ReadConvertWorker():
    def __init__(self, source_queue, dest_queue):
        self.source_queue = source_queue
        self.dest_queue = dest_queue

    def __call__(self):
        while True:
            item = self.source_queue.get()
            if item is None or len(item) != 3:
                self.source_queue.task_done()
                break
            iimg, patient_id, labeling = item
            print('Processing image {}: {}'.format(iimg, patient_id))
            dcm_path = os.path.join(FLAGS.input_images_path, patient_id + '.dcm')
            bboxes = [Bbox(**g) for g in labeling if int(g.get('Target', -1)) > 0]
            tf_example = create_tf_example(dcm_path, bboxes)
            self.dest_queue.put(tf_example)
            self.source_queue.task_done()


class WriteWorker():
    def __init__(self, source_queue, writer):
        self.source_queue = source_queue
        self.writer = writer

    def __call__(self):
        while True:
            tf_example = self.source_queue.get()
            if tf_example is None:
                break
            self.writer.write(tf_example.SerializeToString())
            self.source_queue.task_done()


def patient_key(d):
    return d['patientId']


# In[] Main
def main(_):
# In[]
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    readQueue = queue.Queue()
    writeQueue = queue.Queue()
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # Start threads
    threads = []
    for i in range(FLAGS.threads):
        t = threading.Thread(target=ReadConvertWorker(readQueue, writeQueue))
        t.start()
        threads.append(t)
    t = threading.Thread(target=WriteWorker(writeQueue, writer))
    t.start()
    threads.append(t)

    # Start queue
    with open(FLAGS.input_labeling_path, newline='') as csvfile:
        datareader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        if FLAGS.take_first_n_elements > 0:
            datareader = itertools.islice(datareader,
                                          FLAGS.take_first_n_elements)
        datareader = sorted(datareader, key=patient_key)
        datareader = itertools.groupby(datareader, key=patient_key)
        for iimg, (key, group) in enumerate(datareader):
            readQueue.put((iimg, key, group))

    # block until all tasks are done
    readQueue.join()
    writeQueue.join()

    # stop workers
    for i in range(FLAGS.threads):
        readQueue.put(None)
    writeQueue.put(None)

    for t in threads:
        t.join()
    writer.close()

# In[]
if __name__ == '__main__':
    tf.app.run()