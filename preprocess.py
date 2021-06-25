import os
import matplotlib.pyplot as plt
import tensorflow as tf

import hyperparameters as hp


class Datasets():
    def __init__(self):

        self.train_data, self.val_data = self.get_data(
            os.path.join(hp.DATA_PATH, 'train', hp.DATA_DIR), True)

        self.test_data = self.get_data(
            os.path.join(hp.DATA_PATH, 'val', hp.DATA_DIR), False)

    def get_data(self, path, split):

        train_list = []
        train_labels = []

        for dir in os.listdir(path):

            file_path = os.path.join(path, dir, 'frame_00000000_Color_00.png')
            nocs = os.path.join(path, dir, 'frame_00000000_NOXRayTL_00.png')
            xnocs = os.path.join(path, dir, 'frame_00000000_NOXRayTL_01.png')
            peel = os.path.join(path, dir, 'frame_00000000_Color_01.png')

            train_list.append(file_path)
            train_labels.append([nocs, xnocs, peel])

        dataset = tf.data.Dataset.from_tensor_slices(
            (train_list, train_labels))

        def _parse_function(filename, labelname):
            image_string = tf.io.read_file(filename)
            image_decoded = tf.image.decode_png(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)/255.0

            nocs_string = tf.io.read_file(labelname[0])
            nocs_decoded = tf.image.decode_png(nocs_string, channels=3)
            nocs = tf.cast(nocs_decoded, tf.float32)/255.0
            # mask_nocs = tf.expand_dims(tf.where(tf.norm(nocs, axis=-1)>= 1.7320508, 0.0, 1.0), axis=-1)

            xnocs_string = tf.io.read_file(labelname[1])
            xnocs_decoded = tf.image.decode_png(xnocs_string, channels=3)
            xnocs = tf.cast(xnocs_decoded, tf.float32)/255.0
            # mask_xnocs = tf.expand_dims(tf.where(tf.norm(xnocs, axis=-1)>= 1.7320508, 0.0, 1.0), axis=-1)

            peel_string = tf.io.read_file(labelname[2])
            peel_decoded = tf.image.decode_png(peel_string, channels=3)
            peel = tf.cast(peel_decoded, tf.float32)/255.0

            label = tf.concat([nocs, xnocs, peel], -1)

            return image, label

        dataset = dataset.map(_parse_function)
        # for next_element in dataset:
        #     tf.print(next_element)
        #     plt.imshow(next_element[1][:,:,3].numpy())
        #     plt.show()
        if split:
            train_data = dataset.skip(350).batch(hp.BATCH_SIZE)
            val_data = dataset.take(350).batch(hp.BATCH_SIZE)
            return train_data, val_data
        else:
            dataset = dataset.batch(hp.BATCH_SIZE)
            return dataset
