import numpy as np
import tensorflow as tf

from .data_processor import Processor


class ShiftChannel(Processor):
    """Shift image channel between NCHW and NHWC"""

    def __call__(self, image, **kwargs):
        image = tf.transpose(image, [2, 1, 0])

        return dict({'image': image})


class PerImageStandardization(Processor):
    """Image standardization per image.

    https://www.tensorflow.org/api_docs/python/image/image_adjustments#per_image_standardization

    Args:
        image: An image numpy array.
    """
    def __call__(self, image, **kwargs):
        return dict({'image': tf.image.per_image_standardization(image)})


class Resize(Processor):
    """Resize an Image to size"""

    def __init__(self, size):
        self.size = size

    def __call__(self, image, **kwargs):
        resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        image = tf.image.resize_images(image, tf.convert_to_tensor(self.size), method=resize_method)

        return dict({'image': image})


class ResizeWithGtBoxes(Processor):
    """Resize an image and gt_boxes.

    Args:
        image(np.ndarray): An image numpy array.
        gt_boxes(np.ndarray): Ground truth boxes in the image. shape is [num_boxes, 5(x, y, width, height)].
        size: [height, width]
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt_boxes, **kwargs):

        origin_width = image.shape[1]
        origin_height = image.shape[0]

        width = self.size[1]
        height = self.size[0]

        resizer = Resize(self.size)
        resized_image = resizer(image)

        scale = [height / origin_height, width / origin_width]

        if gt_boxes is not None and len(gt_boxes) != 0:
            gt_boxes[:, 0] = gt_boxes[:, 0] * scale[1]
            gt_boxes[:, 1] = gt_boxes[:, 1] * scale[0]
            gt_boxes[:, 2] = gt_boxes[:, 2] * scale[1]
            gt_boxes[:, 3] = gt_boxes[:, 3] * scale[0]

            # move boxes beyond boundary of image for scaling error.
            gt_boxes[:, 0] = np.minimum(gt_boxes[:, 0], width - gt_boxes[:, 2])
            gt_boxes[:, 1] = np.minimum(gt_boxes[:, 1], height - gt_boxes[:, 3])

        return resized_image, gt_boxes
