import os
import json
import copy
import numpy as np
import tensorflow as tf

from PIL import Image
from collections import OrderedDict


def get_datatype(obj):

    if type(obj) == np.ndarray:
        data_type = str(obj.dtype)
    else:
        data_type = str(type(obj))

    if 'int' in data_type:
        typed_data = 'int'
    elif 'float' in data_type:
        typed_data = 'float'
    elif 'bytes' in data_type:
        typed_data = 'bytes'
    elif 'str' in data_type:
        typed_data = 'str'
    else:
        raise RuntimeError

    return typed_data


def tfrecord_meta_file(tfrecord_file_path):
    return ''.join(str(tfrecord_file_path).split('.tfrecord')[:-1] + ['.meta'])


def tfr_datalist(raw_data, pre_processor=None):
    datalist = []
    total_data = len(raw_data[0])
    count = 0
    print('Generated ... %d %% [%s of %s]' % (((count / total_data) * 100), count, total_data), end='')
    for image_data, labels in zip(*raw_data):
        count += 1
        print('\rGenerated ... %d %% [%s of %s]' % (((count / total_data) * 100), count, total_data), end='')

        feature_dict = OrderedDict()
        img_dtype = get_datatype(image_data)
        if img_dtype == 'bytes':
            image_data = tf.gfile.GFile(image_data, 'rb').read()
        elif img_dtype == 'str':
            image_data = np.asarray(Image.open(image_data).convert('RGB'))
        elif img_dtype == 'int':
            image_data = image_data
            img_dtype = 'bytes'

        image_data = np.reshape(image_data, newshape=[3, 32, 32])

        samples = {'image': image_data, 'labels': labels}

        if callable(pre_processor):
            samples = pre_processor(**samples)

        image_data = samples['image']
        labels = samples['labels']

        feature_data = OrderedDict()
        feature_data['feature/image'] = tf.compat.as_bytes(image_data.tostring())

        del image_data

        if 'feature/image' not in feature_dict.keys():
            feature_dict['feature/image'] = img_dtype

        if type(labels) is list:
            for label in labels:
                label = np.split(np.asarray(label), len(label))
                for i, lbl_item in enumerate(label):
                    if 'feature/label_%s' % i not in feature_dict.keys():
                        lbl_dtype = get_datatype(lbl_item)
                        feature_dict['feature/label_%s' % i] = lbl_dtype
                    feature_data['feature/label_%s' % i] = lbl_item
        else:
            if 'feature/label_0' not in feature_dict.keys():
                lbl_dtype = get_datatype(labels)
                feature_dict['feature/label_0'] = lbl_dtype
            feature_data['feature/label_0'] = labels

        datalist.append(feature_data)

    print()
    return datalist, feature_dict


def write_tfrecords(output_path, data_list, meta_dict):
    """
    Write tfrecords in file

    :param output_path: tfrecord file path
    :param data_list: list of data the corresponding to each feature
    :param meta_dict: dict of the meta information
    :return: extracted data as list of numpy arrays
    """

    writer = tf.python_io.TFRecordWriter(output_path)

    meta_file = tfrecord_meta_file(output_path)

    data_list, feature_dict = data_list

    meta_dict.update(feature_dict.copy())
    with open(meta_file, 'w') as f:
        # f.writelines("total:%s\n" % total_data)
        json.dump(meta_dict, f, indent=4)

    count = 0
    total_data = len(data_list)
    print('Writing TFRecord ... %d %% [%s of %s]' % (((count / total_data) * 100), count, total_data), end='')
    for data in data_list:
        count += 1
        print('\rWriting TFRecord ... %d %% [%s of %s]' % (((count / total_data) * 100), count, total_data), end='')
        tfr_example = encode_tfrecoreds(feature_dict, data)
        writer.write(tfr_example.SerializeToString())
    print()


def encode_tfrecoreds(feature_dict, data_dict):
    """
    Encode data into tfrecors examples

    :param features: dict of the features e.g. { 'feature_name': 'data_type_as_str[int, float, byte]' }
    :param data: value of the corresponding feature
    :return: tf.train.Example
    """

    def int64_feature(value):
        value_list = value if type(value) is list else [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

    def bytes_feature(value):
        value_list = value if type(value) is list else [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))

    def float_feature(value):
        value_list = value if type(value) is list else [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))

    features = {}
    for feature, data_type in feature_dict.items():
        data = data_dict[feature]
        if data_type == 'int':
            typed_data = int64_feature(data)
        elif data_type == 'float':
            typed_data = float_feature(data)
        elif data_type == 'bytes':
            typed_data = bytes_feature(data)
        elif data_type == 'str':
            if isinstance(data, str):
                typed_data = bytes_feature(str.encode(data))
            else:
                typed_data = bytes_feature(data)
        else:
            raise RuntimeError

        features[feature] = copy.deepcopy(typed_data)

    return tf.train.Example(features=tf.train.Features(feature=features))

