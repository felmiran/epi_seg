from skimage import util
import os
from cv2 import imread
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from math import floor




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
                                  tile_height, tile_width, 3)

    return imageCopy


def convert_mask_to_labels(mask, tile_height, tile_width):
    '''
    converts mask to a list of labels. 
    Mask is read as numpy array
    '''
    ver = floor(mask.shape[0])
    hor = floor(mask.shape[1])

    y = [1
         if np.sum(mask[tile_side * ver : tile_height * (ver + 1),
                        tile_side * hor : tile_width * (hor + 1)
                        ]) == tile_side_squared
         else 0
         for ver in range(0, n_ver)
         for hor in range(0, n_hor)]
    return y


def create_filename_list():
    # TODO>
    '''
    creates a dictionary of files from the images in the "split" directory
    '''

    filename_list = os.listdir("data/split/X")
    filename_list.remove(".gitkeep")
    return filename_list


class ImageGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_IDs, batch_size=1, tile_side,
                 shuffle=True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.tile_side = tile_side
        self.n_channels = n_channels
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
        image = imread("data/split/X/" + list_IDs_temp[0])
        mask = imread("data/split/mask/" + list_IDs_temp[0])
        
        X = convert_image_to_stack_of_tiles(image, tile_side, tile_side)
        y = convert_mask_to_labels(mask, tile_side, tile_side)

        return X, y



# def basic_dl_model()




def main():

    '''
    converts image to stack and runs training of DL model by use of custom
    Generator from Keras
    '''
    
    
    
    
    
    
    # print(os.getcwd())
    # image_dir = "data/split/X"
    # filename = "S04_292_p16_RTU_ER1_20 - 2016-04-12 15.42.13(20480,5120)_" + \
    #            "5120x5120.tif"
    # image = imread(image_dir + "/" + filename)
    # tile_height, tile_width = 1280, 1280

    # stacked = convert_image_to_stack_of_tiles(image, tile_height, tile_width)

    # print(stacked.shape)

    # muestra = stacked[5]

    # plt.figure()
    # plt.imshow(muestra)
    # plt.show()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()