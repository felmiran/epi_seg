from skimage import util
import os
from utils import list_files_from_dir, normalize_image, \
    train_validation_test_partition
from metrics import *
from preprocess import data_augmentation
from cv2 import imread, IMREAD_GRAYSCALE
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
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


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, model_name, tile_side):
        self.model_name = model_name
        self.tile_side = tile_side
        super().__init__()
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 1 == 0:
            self.model.save(self.model_name.format(self.tile_side, epoch+1))
            print("class weights saved!")



def apply_distortion(img, tile_side=128, distortion='gauss'):
    'Rotate 90° clockwise'
    if distortion == '90r':
        return tf.image.rot90(img, k=3)

    if distortion == 'color':

        color_dist = np.zeros((tile_side, tile_side, 3))
        color_dist[:,:,1] = 13./255 # con esto vamos a aumentar en 13/255 el canal 1, que corresponde a G (independiente de si es RGB o BGR)
        color_dist_tensor = tf.convert_to_tensor(color_dist, dtype=np.float32)

        return img + color_dist_tensor
    
    if distortion == 'gauss':
        gauss = tf.random.normal(tf.shape(img), 0., .5, )
        return img + gauss
    
    
def custom_loss(i_dist_output_layer, alpha=1.):
    
    def loss(y_true, y_pred):
        l_0 = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # NOTE: keras KL divergence only uses column that was provided (y), so we have to add probabilities of other alternative (1-y)
        # This is because of how kL divergence is calculated
        l_stability_col_1 = tf.keras.losses.kullback_leibler_divergence(y_pred, i_dist_output_layer)
        l_stability_col_2 = tf.keras.losses.kullback_leibler_divergence(1-y_pred, 1-i_dist_output_layer) 
        l_stability = l_stability_col_1 + l_stability_col_2

        # l_stability_print = tf.print('\nl_stability shape is: ', tf.shape(l_stability), '\nl_stability is: ', l_stability, 
        #                              '\nl_0 shape is: ', tf.shape(l_0), '\nl_0: ', l_0, 
        #                              '\ndist_output shape is: ', tf.shape(i_dist_output_layer), '\ndist_output is: ', i_dist_output_layer,
        #                              '\ny_pred shape is: ', tf.shape(y_pred), '\ny_pred is: ', y_pred,
        #                              '\ny_true shape is: ', tf.shape(y_true), '\ny_true is: ', y_true)
        
        # with tf.control_dependencies([l_stability_print]):
        #     return tf.identity(l_stability)

        return l_0 + l_stability * alpha
        

    return loss

def basic_dl_model(tile_side, saver, model_name, training_generator, validation_generator=None,
                   class_weight={0: 1., 1: 1.}, epochs=5):

    # https://www.kdnuggets.com/2019/04/advanced-keras-constructing-complex-custom-losses-metrics.html


    '''modelo 1: modelo mas simple, solo un par de capas convolucionales'''
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
    #                            input_shape=(tile_side, tile_side, 3),
    #                            data_format="channels_last", activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
    #                            data_format="channels_last", activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
    #                            data_format="channels_last", activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid'),
    # ])

    '''modelo 2: solo el modelo base mas un par de capas fully connected'''
    # conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
    conv_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
    # conv_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))

    model = tf.keras.models.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    conv_base.trainable=True


    '''modelo 3: incluye custom loss para aplicar stability training'''
    # i = tf.keras.layers.Input(shape=(tile_side, tile_side, 3))
    # i_dist = tf.keras.layers.Lambda(apply_distortion)(i)

    # inception = tf.keras.applications.InceptionV3(weights='imagenet', 
    #                                             include_top=False, 
    #                                             input_shape=(tile_side,tile_side,3))


    # base_model = tf.keras.models.Sequential()
    # base_model.add(inception)
    # base_model.add(tf.keras.layers.Flatten())
    # base_model.add(tf.keras.layers.Dense(128, activation='relu'))
    # base_model.add(tf.keras.layers.Dense(64, activation='relu'))
    # base_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # x_i = base_model(i)
    # x_i_dist = base_model(i_dist)

    # model = tf.keras.models.Model(inputs=i, outputs=x_i)


    '''compile/fit'''
    model.compile(optimizer='adam', 
                #   loss=custom_loss(x_i_dist),
                  loss='binary_crossentropy',
                  metrics=['acc', precision_m, recall_m, f1_m, auc_m])

    # $ tensorboard --logdir=./
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log/{}'.format(model_name.format(tile_side, "all")))
    
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        callbacks=[saver, tensorboard],
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=8,
                        class_weight=class_weight)
    return model


def InceptionModel(tile_side, training_generator):
    pass


def main():
    '''
    trains a model from a set of tiles previously preprocessed and separated into
    folders according to the label
    '''
    
    model_name = "models/" + time.strftime("%Y%m%d") + \
                     "_InceptionV3_15pics_x20_AUCMetric_{}px_epoch_{}.h5"
                    #  "_InceptionV3_newPreprocesing_model_{}_epochs_15pics_absnorm_x20_{}px_full-train.h5"

    
    file_list, dir_list, counts = list_files_from_dir(directory="data/train/split/X",
                                              extension=".tif")

    print("\n\n\nNumber of elements in file list: " + str(len(file_list)) +"\n\n")

    train_list, val_list, _ = train_validation_test_partition(file_list,
                                                              prop=(0.8,
                                                                    0.2,
                                                                    0.0))
    
    ild = {file_list[i]: dir_list[i] for i in range(len(dir_list))}

    tile_side = 128
    saver = CustomSaver(model_name=model_name, tile_side=tile_side)
    training_generator = ImageGenerator2(list_IDs=train_list,
                                         image_label_directory=ild,
                                         tile_side=tile_side,
                                         batch_size=64)
    validation_generator = ImageGenerator2(list_IDs=val_list,
                                           image_label_directory=ild,
                                           tile_side=tile_side,
                                           batch_size=64)
    
    class_weight = {0: 1., 1: (counts["-1"]+counts["0"])/counts["1"]}    


    
    epochs = [40]
    for e in epochs:
        model = basic_dl_model(tile_side,
                               saver=saver,
                               model_name=model_name,
                               training_generator=training_generator,
                               validation_generator=validation_generator,
                               class_weight=class_weight,
                               epochs=e)

        model.save(model_name.format(tile_side, e))
    
    K.clear_session()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
