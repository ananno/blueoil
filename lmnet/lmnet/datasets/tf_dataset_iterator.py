import os
import shutil
from .base import Base
from lmnet import environment as env
from lmnet.tfrecord_processor import *
from .tfrecord_dataset import TFRDataset


class TFIterator:
    def __init__(self,
                 dataset, config, on_hot=True, reset=True):
        self.config = config
        self.dataset = dataset
        self.subset = dataset.subset
        self.images = dataset.images
        self.labels = dataset.labels
        self.classes = dataset.classes
        self.image_shape = dataset.image_shape
        self.data_format = dataset.data_format
        self.batch_size = dataset.batch_size
        self.num_per_epoch = dataset.num_per_epoch
        self.pre_processor = dataset.pre_processor
        self.augmentor=dataset.augmentor
        self.one_hot = on_hot
        self.reset = reset

        if 'num_max_boxes' in dataset.__dict__.keys():
            self.num_max_boxes = dataset.num_max_boxes

        self.tfrecord_dir = env.TFR_DIR
        self.tfrecord_path = os.path.join(self.tfrecord_dir, dataset.subset + '.tfrecord')

        self.recreate()

    def recreate(self):
        if os.path.exists(self.tfrecord_path):
            shutil.rmtree(self.tfrecord_path, ignore_errors=True)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)

    def get_iterator(self):
        print("Preparing TF_DATASET_ITERATOR for %s ..." % self.subset)
        if not os.path.exists(self.tfrecord_path) or self.reset:
            write_tfrecords(output_path=self.tfrecord_path,
                            data_list=tfr_datalist([self.images, self.labels], self.pre_processor),
                            meta_dict={'meta/count': len(self.images)})
        else:
            print('TFRecord found : %s' % self.tfrecord_path)

        buffer_size = self.batch_size * 3
        tf_dataset = TFRDataset(name=self.subset,
                                tfrecord_path=self.tfrecord_path,
                                config=self.config,
                                dataset=self.dataset,
                                buffer_size=buffer_size,
                                pre_processor=self.pre_processor,
                                augmentor=self.augmentor,
                                one_hot=self.one_hot)

        return tf_dataset

