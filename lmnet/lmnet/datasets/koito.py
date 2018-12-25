# -*- coding: utf-8 -*-
import json
import random
import os.path
import functools
import PIL.Image
import numpy as np
from multiprocessing import Pool, cpu_count

from lmnet.datasets.base import ObjectDetectionBase

DATA_DIR = "/storage/koito_phase3"
IMAGE_DIR = "pics"
JSON_PATH = "shuffled_modified_project_1_1477962836.json"
TRAIN_LIST = "train_list.txt"
TEST_LIST = "test_list.txt"


def fetch_one_data(args):
    image_file, gt_boxes, augmentor, pre_processor, is_train = args
    image = PIL.Image.open(image_file)
    image = np.array(image)
    gt_boxes = np.array(gt_boxes)
    samples = {'image': image, 'gt_boxes': gt_boxes}

    if callable(augmentor) and is_train:
        samples = augmentor(**samples)

    if callable(pre_processor):
        samples = pre_processor(**samples)

    image = samples['image']
    gt_boxes = samples['gt_boxes']

    return (image, gt_boxes)


class Koito(ObjectDetectionBase):
    """Koito dataset for object detection.

        images: images numpy array. shape is [batch_size, height, width]
        labels: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """

    classes = ['leading_car', 'oncoming_car']
    num_classes = len(classes)
    available_subsets = ["train", "validation"]
    extend_dir = 'data'

    @classmethod
    def count_max_boxes(cls, base_path=None):
        """Count max boxes size over all subsets."""
        num_max_boxes = 0

        for subset in cls.available_subsets:
            obj = cls(subset=subset, base_path=base_path)
            gt_boxes_list = obj.labels

            subset_max = max([len(gt_boxes) for gt_boxes in gt_boxes_list])
            if subset_max >= num_max_boxes:
                num_max_boxes = subset_max

        return num_max_boxes

    def __init__(
            self,
            *args,
            **kwargs
    ):
        if "enable_prefetch" in kwargs:
            if kwargs["enable_prefetch"]:
                self.use_prefetch = True
            else:
                self.use_prefetch = False
            del kwargs["enable_prefetch"]
        else:
            self.use_prefetch = False

        super().__init__(
            *args,
            **kwargs,
        )
        self.current_batch_index = 0
        self.random_state = 0

        self.images_path = os.path.join(self.data_dir, IMAGE_DIR)
        self.label_path = os.path.join(self.data_dir, JSON_PATH)
        self.train_list = os.path.join(self.data_dir, TRAIN_LIST)
        self.test_list = os.path.join(self.data_dir, TEST_LIST)

        self._init_files_and_annotations()

        if self.use_prefetch:
            self.enable_prefetch()
            print("ENABLE prefetch")
        else:
            print("DISABLE prefetch")

    def prefetch_args(self, i):
        return self.images[i], self.labels[i], self.augmentor, self.pre_processor, self.subset == "train"

    def enable_prefetch(self):
        # TODO(tokunaga): the number of processes should be configurable
        self.pool = Pool(processes=cpu_count() // 2 if cpu_count() > 1 else 1)
        self.start_prefetch()
        self.use_prefetch = True

    def start_prefetch(self):
        index = self.current_batch_index
        batch_size = self.batch_size
        start = index
        end = min(index + batch_size, self.num_per_epoch)
        pool = self.pool

        args = []
        for i in range(start, end):
            args.append(self.prefetch_args(i))

        self.current_batch_index += batch_size
        if self.current_batch_index >= self.num_per_epoch:
            self.current_batch_index = 0
            self._shuffle()

            rest = batch_size - len(args)
            for i in range(0, rest):
                args.append(self.prefetch_args(i))
            self.current_batch_index += rest

        self.prefetch_result = pool.map_async(fetch_one_data, args)

    @staticmethod
    def _normalize_bboxes(bboxes, image_shape, normalizer=None):
        if normalizer is None:
            h = image_shape[0]
            w = image_shape[1]
        else:
            h, w = normalizer

        x, y, ws, hs, c = np.split(np.array(bboxes), 5, axis=-1)
        x1, y1, x2, y2 = x, y, x + ws, y + hs
        y1 = np.array(y1 / h, dtype=np.float32)
        x1 = np.array(x1 / w, dtype=np.float32)
        y2 = np.array(y2 / h, dtype=np.float32)
        x2 = np.array(x2 / w, dtype=np.float32)

        result = np.concatenate([x1, y1, x2 - x1, y2 - y1, c], axis=-1)

        return result

    @staticmethod
    def _denormalize_bboxes(bboxes, image_shape, normalizer=None):
        if normalizer is None:
            w = image_shape[0]
            h = image_shape[1]
        else:
            h, w = normalizer

        w = float(w)
        h = float(h)

        try:
            y1, x1, y2, x2, c = np.split(np.array(bboxes, dtype=np.float32), 5, axis=-1)
            x1 *= w
            x2 *= w
            y1 *= h
            y2 *= h

            result = np.concatenate([y1, x1, y2, x2, c], axis=-1)

        except Exception as excp:
            result = [[0, 0, 0, 0, -1]]
            print(excp)

        return result

    @property
    def num_per_epoch(self):
        return len(self.images)

    def _element(self):
        """Return an image, gt_boxes."""
        index = self.current_element_index

        self.current_element_index += 1
        if self.current_element_index == self.num_per_epoch:
            self.current_element_index = 0
            self._shuffle()

        files, gt_boxes_list = self.images, self.labels
        target_file = files[index]
        gt_boxes = gt_boxes_list[index]
        gt_boxes = np.array(gt_boxes)

        image = PIL.Image.open(target_file)
        image = np.array(image)

        samples = {'image': image, 'gt_boxes': gt_boxes}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        gt_boxes = samples['gt_boxes']

        return image, gt_boxes

    @staticmethod
    def _read_txt_data(file_path):
        data = []
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                data.append(line.split(' '))

        return data

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _load_json(json_file):
        f = open(json_file)
        data = json.load(f)
        f.close()

        return data

    def _load_image_and_labels(self, image_path, label_path):
        assert os.path.exists(image_path), image_path
        assert os.path.exists(label_path), label_path

        def read_ans_json(json_filepath, img_dir="", useless_category_ids=[]):
            """
            jsonファイル読み込み
            """

            def calc_category_convert(loaded_json, useless_category_ids):
                """
                Suppose catogories are [1, 2, 3, 4, 5]
                and useless_category_ids are [2, 4]
                return will be {1: 1, 3: 2, 5: 3}
                """
                category_ids = [k["id"] for k in loaded_json["categories"]]
                convert_dict = {}

                new_category_id = 0
                for current_category_id in category_ids:
                    if current_category_id not in useless_category_ids:
                        convert_dict[current_category_id] = new_category_id
                        new_category_id += 1

                return convert_dict

            def shift_category_id(all_labels):
                ret = []
                for labels in all_labels:
                    ret_label = []
                    for label in labels:
                        ret_label.append([label[1],
                                    label[2],
                                    label[3],
                                    label[4],
                                    label[0]])
                    ret.append(ret_label)

                return ret

            f = open(json_filepath)
            data = json.load(f)
            f.close()

            ret_images = []
            ret_labels = []

            convert_dict = calc_category_convert(data, useless_category_ids)

            annotations = {}
            for annotation in data["annotations"]:
                annotations[annotation["image_id"]] = []

            for annotation in data["annotations"]:
                if annotation["category_id"] not in useless_category_ids and \
                        annotation['category_id'] in list(convert_dict.keys()):
                    tmp = [convert_dict[annotation["category_id"]],
                           annotation["bbox"][0],
                           annotation["bbox"][1],
                           annotation["bbox"][2],
                           annotation["bbox"][3]]
                    annotations[annotation["image_id"]].append(tmp)

            per_cls_count = {}
            for image in data["images"]:
                if image["id"] in annotations:
                    lbls = annotations[image['id']].copy()
                    for j, lbl in enumerate(lbls):
                        cls = self.classes[lbl[0]]
                        if lbl[3] == 0 or lbl[4] == 0 or lbl[3] <= 30 or lbl[4] <= 30:
                            lbls.pop(j)
                        else:
                            if cls in list(per_cls_count.keys()):
                                per_cls_count[cls] += 1
                            else:
                                per_cls_count[cls] = 1
                    if len(lbls) > 0:
                        ret_images.append(os.path.join(img_dir, image["file_name"]))
                        ret_labels.append(lbls)

            return ret_images, shift_category_id(ret_labels), per_cls_count

        images, labels, per_cls_count = read_ans_json(label_path, image_path)

        return images, labels, per_cls_count

    def train_test_split(self, images, labels, train_filter, test_filter):
        train_data, train_labels, test_data, test_labels = [], [], [], []

        training_list = open(train_filter, 'r').read().split('\n')
        testing_list = open(test_filter, 'r').read().split('\n')

        for i, (img, lbls) in enumerate(zip(images, labels)):
            if os.path.basename(img) in training_list:
                train_data.append(img)
                train_labels.append(lbls)
            elif os.path.basename(img) in testing_list:
                dest_path = os.path.join(self.data_dir, 'test_images', os.path.basename(img))
                #shutil.copyfile(img, dest_path)
                test_data.append(img)
                test_labels.append(lbls)

        return train_data, train_labels, test_data, test_labels

    @functools.lru_cache(maxsize=None)
    def _images_and_annotations(self):
        """Return all files and labels list."""
        single_split_rate = 0.1
        images, labels, per_cls_count = self._load_image_and_labels(self.images_path, self.label_path)

        assert len(images) == len(labels)
        assert len(labels) > 0

        train_files, train_labels, test_files, test_labels = \
            self.train_test_split(images=images,
                                  labels=labels,
                                  train_filter=self.train_list,
                                  test_filter=self.test_list)

        if self.subset == "train":
            images = train_files
            labels = train_labels
        else:
            images = test_files
            labels = test_labels

        print(per_cls_count)
        print("files and annotations are ready")
        return images, labels

    @property
    @functools.lru_cache(maxsize=None)
    def num_max_boxes(self):
        return type(self).count_max_boxes(self.base_path)

    def _one_data(self):
        """Return an image, gt_boxes."""
        index = self.current_batch_index

        self.current_batch_index += 1
        if self.current_batch_index == self.num_per_epoch:
            self.current_batch_index = 0
            self._shuffle()

        files, gt_boxes_list = self.images, self.labels
        target_file = files[index]
        gt_boxes = gt_boxes_list[index]

        gt_boxes = np.array(gt_boxes)

        image = PIL.Image.open(target_file)
        image = np.array(image)

        samples = {'image': image, 'gt_boxes': gt_boxes}

        if callable(self.augmentor) and self.subset == "train":
            samples = self.augmentor(**samples)

        if callable(self.pre_processor):
            samples = self.pre_processor(**samples)

        image = samples['image']
        gt_boxes = samples['gt_boxes']

        return (image, gt_boxes)

    def get_data(self):
        if self.use_prefetch:
            data_list = self.prefetch_result.get(None)
            images, gt_boxes_list = zip(*data_list)
            return images, gt_boxes_list
        else:
            images, gt_boxes_list = zip(*[self._one_data() for _ in range(self.batch_size)])
            return images, gt_boxes_list

    def feed(self):
        """Batch size numpy array of images and ground truth boxes.

        Returns:
          images: images numpy array. shape is [batch_size, height, width]
          gt_boxes_list: gt_boxes numpy array. shape is [batch_size, num_max_boxes, 5(x, y, w, h, class_id)]
        """
        images, gt_boxes_list = self.get_data()

        if self.use_prefetch:
            self.start_prefetch()

        images = np.array(images)
        gt_boxes_list = self._change_gt_boxes_shape(gt_boxes_list)

        if self.data_format == "NCHW":
            images = np.transpose(images, [0, 3, 1, 2])

        return images, gt_boxes_list

    def _shuffle(self):
        """Shuffle data if train."""

        if self.subset == "train":
            # self.files, self.annotations = sklearn.utils.shuffle(
            #     self.files, self.annotations, random_state=self.random_state)
            zipped_list = list(zip(*(self.images, self.labels)))
            random.shuffle(zipped_list)
            self.images, self.labels = zip(*zipped_list)
            print("Shuffle {} train dataset with random state {}.".format(self.__class__.__name__, self.random_state))
            self.random_state = self.random_state + 1

    def _init_files_and_annotations(self):
        self.images, self.labels = self._images_and_annotations()
        self._shuffle()
