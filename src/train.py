from skimage import util
import os
from cv2 import imread
import numpy as np
from matplotlib import pyplot as plt


def SLIC():

    # TODO
    '''
    crear los tiles mediante SLIC

    '''

    pass


def convert_to_stack_of_tiles(image, tile_height, tile_width):
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


def main():

    '''
    converts image to stack and runs training of DL model by use of custom
    Generator from Keras
    '''
    print(os.getcwd())
    image_dir = "data/split/X"
    filename = "S04_292_p16_RTU_ER1_20 - 2016-04-12 15.42.13(20480,5120)_" + \
               "5120x5120.tif"
    image = imread(image_dir + "/" + filename)
    tile_height, tile_width = 1280, 1280

    stacked = convert_to_stack_of_tiles(image, tile_height, tile_width)

    print(stacked.shape)

    muestra = stacked[5]

    plt.figure()
    plt.imshow(muestra)
    plt.show()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
