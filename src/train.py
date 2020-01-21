from skimage import util
import os
from utils import list_files_from_dir, normalize_image, \
    train_validation_test_partition, to_hsv, blur_image
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
import json


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

# class ImageGenerator1(tf.keras.utils.Sequence):
#     def __init__(self, list_IDs, tile_side, batch_size=1,
#                  shuffle=True):
#         self.list_IDs = list_IDs
#         self.batch_size = batch_size
#         self.tile_side = tile_side
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         # line below is commented as batch size is 1 for this problem.
#         # return int(np.floor(len(self.list_IDs) / self.batch_size))
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         list_IDs_temp = [self.list_IDs[index]]
#         X, y = self.__data_generation(list_IDs_temp)
#         return X, y

#     def on_epoch_end(self):
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         '''
#         generates batches of shape:
#         (n_samples, tile_side, tile_side, n_channels)

#         For now it might not be necesary, but it will be when we want to
#         preprocess the images before feeding them.

#         input:
#          - list_IDs_temp: list with image filenames. For now, it only consists
#            on a list with one element
#         '''
#         image = imread("data/train/split/X/" + list_IDs_temp[0])
#         image = normalize_image(image)
#         mask = imread("data/train/split/mask/" + list_IDs_temp[0],
#                       IMREAD_GRAYSCALE)

#         X = convert_image_to_stack_of_tiles(image, self.tile_side,
#                                             self.tile_side)
#         y = convert_mask_to_labels(mask, self.tile_side,
#                                    self.tile_side)

#         # print("Shape of X: " + str(X.shape))
#         # print("Length of y: " + str(len(y)))

#         return shuffle(X, y)


class ImageGenerator2(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, image_label_directory, source_directory, tile_side=128,
                 batch_size=100, shuffle=True, hsv=False):
        self.source_directory = source_directory
        self.list_IDs = list_IDs
        self.image_label_directory = image_label_directory
        self.tile_side = tile_side
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hsv = hsv
        np.random.seed(24)
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

        input:
         - list_IDs_temp: list with image filenames. For now, it only consists
           on a list with one element
        '''

        X = np.empty((self.batch_size, self.tile_side, self.tile_side, 3), dtype='float32')
        y = np.empty((self.batch_size), dtype=int)

        for i, fname in enumerate(list_IDs_temp):
            X[i] = imread(self.source_directory + "/" + self.image_label_directory[fname] + "/" + fname)
            if self.hsv==True:
                X[i] = to_hsv(X[i])
            X[i] = normalize_image(image=X[i], hsv=self.hsv)
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



def apply_distortion(img, tile_side=128, distortion='blur'):
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

    if distortion == 'blur':
        # placeholder = tf.placeholder(tf.float32, shape=[1, tile_side, tile_side, 3])
        return blur_image(img)


def apply_distortion_2_tensor(norm_image_tensor, color=False, saturation=False, contrast=False, brightness=False, gauss=False):
    tf_image = norm_image_tensor
    if color:
        color_modifier = np.zeros((128,128,3), dtype='int8')
        color_modifier[:,:,0] += np.random.randint(-10,10)
        color_modifier[:,:,1] += 0
        color_modifier[:,:,2] += np.random.randint(-10,10)
        color_modifier = tf.convert_to_tensor(color_modifier/255*2, dtype=np.float32)
        tf_image = tf_image + color_modifier

    if saturation:
        # tf_image = tf.image.random_saturation(tf_image, 1., 5.)
#         tf_image = tf.image.adjust_saturation(tf_image, 2)
        pass
    if contrast:
        tf_image = tf.image.random_contrast(tf_image, .8, 1.2)
#         tf_image = tf.image.adjust_contrast(tf_image, .8)
        
    if brightness:
        tf_image = tf.image.random_brightness(tf_image, .3)
    
    if gauss:
#         tf_image = normalize_image(tf_image)
#         gauss = (tf.random.normal(tf.shape(tf_image), 0, .5))
#         tf_image = tf_image+gauss
#         return denormalize_image(tf_image.eval(session=tf.Session()))
        pass

    return tf.clip_by_value(tf_image, -1, 1)
    
    
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
        #     # return tf.identity(l_stability)
        #     return tf.identity(l_0 + l_stability * alpha)

        return l_0 + l_stability * alpha
        

    return loss

def basic_dl_model(tile_side, saver, model_name, training_generator, validation_generator=None,
                   class_weight={0: 1., 1: 1.}, epochs=5, base="basic", trainable=True):

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
    
    # if base == "Xception":
    #     conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
    # elif base == "InceptionV3":
    #     conv_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
    # elif base == "InceptionResNetV2":
    #     conv_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
    # elif base == "ResNet50":
    #     conv_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
    # else:
    #     conv_base = tf.keras.models.Sequential([
    #             tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), input_shape=(tile_side, tile_side, 3), data_format="channels_last", activation='relu'),
    #             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #             tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
    #             tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
    #             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #             tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
    #             tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
    #             tf.keras.layers.MaxPooling2D(pool_size=(2, 2))])

    # model = tf.keras.models.Sequential()
    # model.add(conv_base)
    # model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # conv_base.trainable=trainable




    '''modelo 3: incluye custom loss para aplicar stability training'''
    i = tf.keras.layers.Input(shape=(tile_side, tile_side, 3))
    i_dist = tf.keras.layers.Lambda(apply_distortion_2_tensor, arguments={'color': True, 'contrast': True, 'brightness': True})(i)

    inception = tf.keras.applications.InceptionV3(weights='imagenet', 
                                                include_top=False, 
                                                input_shape=(tile_side,tile_side,3))


    base_model = tf.keras.models.Sequential()
    base_model.add(inception)
    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Dense(128, activation='relu'))
    base_model.add(tf.keras.layers.Dense(64, activation='relu'))
    base_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    x_i = base_model(i)
    x_i_dist = base_model(i_dist)

    model = tf.keras.models.Model(inputs=i, outputs=x_i)


    '''compile/fit'''
    
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-4/epochs), 
                  loss=custom_loss(x_i_dist),
                  metrics=['acc', precision_m, recall_m, f1_m, auc_m, 'binary_crossentropy', custom_loss(x_i_dist)])
                #   loss='binary_crossentropy',
                #   metrics=['acc', precision_m, recall_m, f1_m, auc_m, 'binary_crossentropy'])

    # $ tensorboard --logdir=./
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log/{}'.format(model_name.format(tile_side, "all")))
    
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  callbacks=[saver, tensorboard],
                                  epochs=epochs,
                                  use_multiprocessing=True,
                                  workers=8,
                                  class_weight=class_weight)
    return model, history


def main():
    '''
    trains a model from a set of tiles previously preprocessed and separated into
    folders according to the label
    '''
    ###  ["modeRange", "1", "InceptionV3", True, "BCE", "RGB", "p16", "15pics", "SGD"]
    ###  [     0        1         2          3     4      5      6        7       8  ]

    ### 0: type of background filterinf --> "stdDev" or "modeRange"
    ### 1: labelling of partially annotated triles --> "0" (non-epithelium) or "1" (epithelium)
    ### 2: DL base architecture --> "InceptionResNetV2", "InceptionV3", "ResNet50", "Xception", "basic" (which is default)
    ### 3: Variable that determines if layers of base architecture are trainable (no transfer learning) or not (transfer learning) --> True, False
    ### 4: Loss function --> "BCE" (binary crossentropy) or "ST" (stability training) (TODO: todavia no esta implementado este parametro, falta terminarlo)
    ### 5: Color model --> "RGB" or "HSV" 
    ### 6: IHC used --> "p16" or "p16+CD8"
    ### 7: images used --> "15pics" or "Box1-4" 
    ### 8: optimizer --> "Adam" o "SGD" (TODO: todavia no esta implemenbtado este parametro)

    parameter_sets = [
                    #   ["modeRange", "1", "InceptionV3", True, "BCE", "HSV", "p16", "15pics", "Adam"],
                    #   ["stdDev", "1", "InceptionV3", True, "BCE", "RGB", "p16", "15pics", "Adam"],
                    #   ["modeRange", "0", "InceptionV3", True, "BCE", "RGB", "p16+CD8", "Box1-4", "SGD"],
                      ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "15pics", "SGD"],
                    #   ["stdDev", "0", "InceptionV3", True, "BCE", "RGB", "p16", "15pics", "Adam"],
                    #   ["modeRange", "0", "InceptionV3", False, "BCE", "RGB", "p16", "15pics", "Adam"],
                    #   ["modeRange", "1", "InceptionV3", False, "BCE", "RGB", "p16", "15pics", "Adam"],
                      ]

    for p_set in parameter_sets:


        name_base = "models/" + time.strftime("%Y%m%d") + "_" + p_set[2] + "_loss" + p_set[4] + \
            "_color" + p_set[5] + "_" + p_set[7] + "_x20_" + p_set[0] + "Prep_partialsAre" + \
                p_set[1] + "_" + p_set[6] + "_opt" + p_set[8] + "_wDropout"

        model_name = name_base + "_{}px_e{}.h5"

        bg_filter_dir = "128px_x20_RGB_{}_{}Preprocessing_partialsAre{}_{}/".format(p_set[7], p_set[0], p_set[1], p_set[6])
        full_dir = "data/train/split/{}X".format(bg_filter_dir)
        file_list, dir_list, counts = list_files_from_dir(directory=full_dir,
                                                          extension=".tif")
        
        tohsv = False
        if p_set[5] == "HSV":
            tohsv = True


        print("\n\nImage folder: " + bg_filter_dir)
        print("Image path: " + full_dir)
        print("Number of elements in file list: " + str(len(file_list)))
        print(str(counts) +"\n\n")
        print("Base model: " + p_set[2])
        print("Base model trainable: " + str(p_set[3]))
        print("optimizer: " + p_set[8])
        print("Loss function: " + str(p_set[4]))
        print("Color model: " + str(p_set[5]) + "- tohsv=" + str(tohsv))

        train_list, val_list, _ = train_validation_test_partition(file_list, prop=(0.6, 0.4, 0.0))
        
        ild = {file_list[i]: dir_list[i] for i in range(len(dir_list))}

        tile_side = 128
        saver = CustomSaver(model_name=model_name, tile_side=tile_side)

        training_generator = ImageGenerator2(list_IDs=train_list,
                                             image_label_directory=ild,
                                             source_directory=full_dir,
                                             tile_side=tile_side,
                                             batch_size=64, 
                                             hsv=tohsv)
        validation_generator = ImageGenerator2(list_IDs=val_list,
                                               image_label_directory=ild,
                                               source_directory=full_dir,
                                               tile_side=tile_side,
                                               batch_size=64,
                                               hsv=tohsv)
        
        class_weight = {0: 1., 1: (counts["-1"]+counts["0"])/counts["1"]}    


        
        # epochs = [40]
        # for e in epochs:
        model, history = basic_dl_model(tile_side,
                                        saver=saver,
                                        model_name=model_name,
                                        training_generator=training_generator,
                                        validation_generator=validation_generator,
                                        class_weight=class_weight,
                                        epochs=40,
                                        base=p_set[2],
                                        trainable=p_set[3])

            # model.save(model_name.format(tile_side, e))
        
        history_name = name_base + "_" + str(tile_side) + "px.json"
        with open(history_name, 'w') as jsonfile:
            json.dump(history.history, jsonfile)

        K.clear_session()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
