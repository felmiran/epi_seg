from skimage import util
import os
from utils import list_files_from_dir, normalize_image, \
    train_validation_test_partition
from metrics import *
from cv2 import imread, IMREAD_GRAYSCALE
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
# from tensorflow.keras import metrics
from math import floor
from sklearn.utils import shuffle
import time


# TODO> rename to "preprocess.py", y crear un nuevo archivo
#       "train.py" que sea el que ejecuta el entrenamiento.


def SLIC():

    # TODO
    '''
    crear los tiles mediante SLIC

    '''

    pass


def convert_image_to_stack_of_tiles(image, tile_height, tile_width):
    '''
    Converts an image to a stack of tiles to be fed to the DL models

    Observations:
    - if (image.shape % tile_height != 0) or (image.shape % tile_width != 0)
      the right and bottom remainder will be cropped so that
      (image.shape % tile_height == 0) and (image.shape % tile_width == 0)

    input:
    - image: the image that will be converted as a numpy array
    - tile_height:
    - tile_width:


    Output:
    np.array((n, h, w, c)),
    n: N° of tiles
    h: tile height
    w: tile width
    c: N° of channels (always 3, as rgb and hsv are used)

    '''
    shape = image.shape

    imageCopy = image[:shape[0] - shape[0] % tile_height,
                      :shape[1] - shape[1] % tile_width,
                      :3]
    imageCopy = util.view_as_blocks(imageCopy, block_shape=(tile_height,
                                                            tile_width, 3))
    imageCopy = imageCopy.reshape(shape[0]//tile_height * shape[1]//tile_width,
                                  tile_height,
                                  tile_width,
                                  3)

    return imageCopy


def convert_mask_to_labels(mask, tile_height, tile_width):
    '''
    converts mask to a list of labels.
    Mask is read as numpy array
    '''
    n_ver = floor(mask.shape[0]/tile_height)
    n_hor = floor(mask.shape[1]/tile_width)

    y = [1
         if np.sum(mask[tile_height * ver:tile_height * (ver + 1),
                        tile_width * hor:tile_width * (hor + 1)
                        ]) == tile_height * tile_width
         else 0
         for ver in range(0, n_ver)
         for hor in range(0, n_hor)]
    return y


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class ImageGenerator1(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, tile_side, batch_size=1,
                 shuffle=True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.tile_side = tile_side
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # line below is commented as batch size is 1 for this problem.
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return len(self.list_IDs)

    def __getitem__(self, index):
        list_IDs_temp = [self.list_IDs[index]]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''
        generates batches of shape:
        (n_samples, tile_side, tile_side, n_channels)

        For now it might not be necesary, but it will be when we want to
        preprocess the images before feeding them.

        input:
         - list_IDs_temp: list with image filenames. For now, it only consists
           on a list with one element
        '''
        image = imread("data/train/split/X/" + list_IDs_temp[0])
        image = normalize_image(image)
        mask = imread("data/train/split/mask/" + list_IDs_temp[0],
                      IMREAD_GRAYSCALE)

        X = convert_image_to_stack_of_tiles(image, self.tile_side,
                                            self.tile_side)
        y = convert_mask_to_labels(mask, self.tile_side,
                                   self.tile_side)

        # print("Shape of X: " + str(X.shape))
        # print("Length of y: " + str(len(y)))

        return shuffle(X, y)


class ImageGenerator2(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, image_label_directory, tile_side=128,
                 batch_size=100, shuffle=True):
        self.list_IDs = list_IDs
        self.image_label_directory = image_label_directory
        self.tile_side = tile_side
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        # return len(self.list_IDs)

    def __getitem__(self, i):
        # generate indexes of the batch
        indexes = self.indexes[i*self.batch_size: (i+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''
        generates batches of shape:
        (n_samples, tile_side, tile_side, n_channels)

        For now it might not be necesary, but it will be when we want to
        preprocess the images before feeding them.

        input:
         - list_IDs_temp: list with image filenames. For now, it only consists
           on a list with one element
        '''

        X = np.empty((self.batch_size, self.tile_side, self.tile_side, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, fname in enumerate(list_IDs_temp):
            X[i] = normalize_image(imread("data/train/split/X/" +
                                          self.image_label_directory[fname] +
                                          "/" + fname))
            y[i] = int(self.image_label_directory[fname].replace("-1", "0"))

        # print("Shape of X: " + str(X.shape))
        # print("Length of y: " + str(len(y)))

        return X, y


def basic_dl_model(tile_side, training_generator, validation_generator=None,
                   epochs=5):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               input_shape=(tile_side, tile_side, 3),
                               data_format="channels_last", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               data_format="channels_last", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               data_format="channels_last", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               data_format="channels_last", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['acc', precision_m,
                           recall_m, f1_m])

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=8,
                        class_weight={0: 1.,
                                      1: 1.3})
    return model


def InceptionModel(tile_side, training_generator):
    pass


def main():
    '''
    converts image to stack and runs training of DL model by use of custom
    Generator from Keras
    '''

    file_list, dir_list = list_files_from_dir(directory="data/train/split/X",
                                              extension=".tif")

    print(len(file_list))
    print(file_list[:10])

    train_list, val_list, _ = train_validation_test_partition(file_list,
                                                              prop=(0.8,
                                                                    0.2,
                                                                    0.0))
    ild = {file_list[i]: dir_list[i] for i in range(len(dir_list))}

    tile_side = 128
    training_generator = ImageGenerator2(list_IDs=train_list,
                                         image_label_directory=ild,
                                         tile_side=tile_side,
                                         batch_size=128)

    validation_generator = ImageGenerator2(list_IDs=val_list,
                                           image_label_directory=ild,
                                           tile_side=tile_side,
                                           batch_size=128)


    epochs = [5, 20, 50]

    for e in epochs:
        model = basic_dl_model(tile_side,
                            training_generator=training_generator,
                            validation_generator=validation_generator,
                            epochs=e)

        model.save("models/" + time.strftime("%Y%m%d-%H%M") +
                "_basic_dl_model_{}_epochs_9pics_absnorm.h5".format(e))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
