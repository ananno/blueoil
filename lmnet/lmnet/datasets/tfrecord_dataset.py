import os
import json
import tensorflow as tf
import multiprocessing as mp
import lmnet.environment as env

from .base import Base
from collections import OrderedDict
from lmnet.tfrecord_processor import tfrecord_meta_file

project_name = "blueoil"


def app_cache_dir(tag='debug'):
    cache_dir = os.path.join(os.path.expanduser("~/.cache"), project_name, tag)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def app_data_cache_dir(tag='debug'):
    data_cache_dir = os.path.join(app_cache_dir(tag), 'data')
    if not os.path.isdir(data_cache_dir):
        os.makedirs(data_cache_dir)
    return data_cache_dir


class TFRDataset(Base):

    def __init__(self,
                 name,
                 tfrecord_path,
                 config,
                 dataset,
                 parsing_fn=None,
                 buffer_size=None,
                 padded_shapes=None,
                 cache_path=None,
                 pre_processor=None,
                 augmentor=None,
                 one_hot=False):

        self.name = name
        self.config = config
        self.dataset = dataset
        self.augmentor = augmentor
        self.pre_processor = pre_processor
        self.data_count = len(dataset.images)
        self.one_hot = one_hot

        self.iterator = None
        self.batch_counter = 0
        self.epoch_counter = 0
        self.initlialized = False
        self.features = None
        self.cache_dir_path = None

        print("Preparing dataset ...")

        meta_file = tfrecord_meta_file(tfrecord_path)
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        feature_dict = OrderedDict()
        for key, value in meta_data.items():
            if str(key).startswith('meta'):
                if key == 'meta/count':
                    self.__data_count = value
            elif str(key).startswith('feature'):
                feature_dict[key] = value

        self.features = feature_dict

        if not isinstance(tfrecord_path, str):
            raise ValueError

        #  os.path.join(tempfile.gettempdir(), os.path.basename(tfrecord_path) + '.cache')

        self.batch_counter = 0
        self.epoch_counter = 0

        # Dataset
        files = tf.data.Dataset.list_files(tfrecord_path)
        dataset = files.interleave(tf.data.TFRecordDataset,
                                   cycle_length=mp.cpu_count())
        # dataset = tf.data.Dataset.from_tensor_slices(files)

        if parsing_fn is None:
            parsing_fn = self.parse_data

        dataset = dataset.map(parsing_fn, num_parallel_calls=mp.cpu_count())

        if cache_path is None:
            cache_path = ''
            # cache_path = os.path.join(app_data_cache_dir(env.EXPERIMENT_ID),
            #                           os.path.splitext(os.path.basename(tfrecord_path))[0])
            #
            # if os.path.exists(cache_path + '.lockfile'):
            #     var = input("Another training might be running ... Do you want to remove the lockfile? [Y/n]")
            #     if var == 'Y' or var == 'y':
            #         os.remove(cache_path + '.lockfile')
            #     else:
            #         exit(1)

        self.cache_dir_path = cache_path

        if buffer_size is None:
            buffer_size = config.BATCH_SIZE

        if cache_path is not '':
            dataset = dataset.cache(cache_path)
        dataset = dataset.prefetch(buffer_size=buffer_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat()
        if padded_shapes is None:
            dataset = dataset.batch(config.BATCH_SIZE)
        else:
            dataset = dataset.padded_batch(config.BATCH_SIZE, padded_shapes=padded_shapes)

        self.tf_dataset = dataset
        self.iterator = self.tf_dataset.make_initializable_iterator()

    def parse_data(self, example_proto):
        encoded_features = {}
        for key, value in self.features.items():
            if value == 'bytes' or value == 'str':
                encoded_features[key] = tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="")
            elif value == 'float':
                encoded_features[key] = tf.FixedLenFeature(shape=(), dtype=tf.float32, default_value=0.0)
            elif value == 'int':
                encoded_features[key] = tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=0)

        parsed_features = tf.parse_single_example(example_proto, encoded_features)
        image_key = 'feature/image'
        image_data = parsed_features[image_key]

        image = tf.decode_raw(image_data, out_type=tf.float32)
        image_shape = self.dataset.image_shape
        image = tf.reshape(image, shape=image_shape)
        if self.config.DATA_FORMAT == 'NHWC':
            image = tf.transpose(image, [1, 2, 0])

        labels = []
        for key, value in parsed_features.items():
            if str(key).startswith('feature') and key != 'feature/image':
                labels.append(value)

        if len(labels) > 1:
            labels = tf.stack(labels)
        else:
            labels = labels[0]

        samples = {'image': image}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        # if callable(self.pre_processor):
        #     samples = self.pre_processor(**samples)

        image = samples['image']

        if self.one_hot:
            labels = tf.one_hot(labels, self.num_classes)

        return image, labels

    def get_next(self):
        if self.data_count is not None:
            if self.batch_counter+1 > self.data_count // self.config.BATCH_SIZE:
                self.batch_counter = 0
                if self.config.max_epochs is not 0 and self.epoch_counter+1 > self.config.max_epochs:
                    raise StopIteration
                self.epoch_counter += 1
            else:
                self.batch_counter += 1
        else:
            self.batch_counter += 1

        try:
            images, labels = self.iterator.get_next()
        except tf.errors.OutOfRangeError:
            return None, None
        except Exception:
            raise Exception

        return images, labels

    def remove_cache(self):
        if self.cache_dir_path and os.path.exists(self.cache_dir_path + '.lockfile'):
            os.remove(self.cache_dir_path + '.lockfile')

    @property
    def subset(self):
        return self.name

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def num_classes(self):
        return len(self.dataset.classes)

    @property
    def extend_dir(self):
        return self.dataset.extend_dir

    @property
    def available_subsets(self):
        return self.dataset.available_subsets

    @property
    def num_per_epoch(self):
        return self.dataset.num_per_epoch

    def feed(self):
        return self.get_next()
