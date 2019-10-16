import os

from train import *
from preprocess import *
from metrics import *
from utils import normalize_image
import tensorflow as tf
import numpy as np
from cv2 import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# TODO>
def evaluate_model(model, ndpi_list):
    '''
    tests images from a list of ndpi filenames to a saved model
    '''
    pass

# Load model
print(os.getcwd())
model_filename = "20191014-1225_basic_dl_model_40_epochs_9pics_absnorm_x20_64px.h5"
# model = tf.keras.models.load_model("src/models/" + model_filename)
model = tf.keras.models.load_model("src/models/" + model_filename, 
                                   custom_objects={'precision_m': precision_m,
                                                   'recall_m': recall_m,
                                                   'f1_m': f1_m})
# print(model.get_weights())


# Evaluate model on data
# ndpi_file = "src/data/test/prueba2.ndpi"
# ndp_image, image_annotation_list = call_ndpi_ndpa(ndpi_file)

# val_X = np.array(ndp_image.read_region(location=(0, 0),
#                                        level=0,
#                                        size=(ndp_image.width_lvl_0,
#                                              ndp_image.height_lvl_0)))
# val_X = cv2.cvtColor(val_X[:, :, :3], cv2.COLOR_RGB2BGR)
tile_size = 64
resolution = "x20"

image_dir = "src/data/test/grandes_RGB/{}".format(resolution)

file_list, _, _ = list_files_from_dir(directory=image_dir, extension=".tif")

# image_filename = "S04_2819_p16_RTU_ER1_20 - 2016-04-12 15.39.25_(30720,16384)_10240x8192.tif"

for image_filename in file_list:
    print(image_filename)

    val_X = cv2.imread(image_dir + "/" + image_filename)

    val_X = normalize_image(val_X)

    n_ver = floor(val_X.shape[0] / tile_size)
    n_hor = floor(val_X.shape[1] / tile_size)

    val_X = convert_image_to_stack_of_tiles(val_X, tile_size, tile_size)


    # val_X = np.load("src/data/test/val_X.npy")

    # n_ver = floor(17152/128)
    # n_hor = floor(14336/128)

    results = model.predict_classes(val_X)
    results = results.reshape(n_ver, n_hor)
    
    # plt.imshow(results, aspect='auto')
    # plt.show()

    save_np_as_image(results*255, image_dir + "/" + image_filename + "_40_epochs_9pics_{}_{}px.png".format(resolution, tile_size))




