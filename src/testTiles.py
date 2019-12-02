import os

# from train import convert_image_to_stack_of_tiles
# from preprocess import *
from metrics import precision_m, recall_m, f1_m
from utils import normalize_image, list_files_from_dir, save_np_as_image
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from cv2 import *
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# TODO>
def evaluate_model(model, ndpi_list):
    '''
    tests images from a list of ndpi filenames to a saved model
    '''
    pass


class ImageTestGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, image_label_directory, tile_side=128,
                 batch_size=100):
        self.list_IDs = list_IDs
        self.image_label_directory = image_label_directory
        self.tile_side = tile_side
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))
        # return len(self.list_IDs)

    def __getitem__(self, i):
        # generate indexes of the batch
        indexes = self.indexes[i*self.batch_size: (i+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

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

        for i, fname in enumerate(list_IDs_temp):
            X[i] = normalize_image(imread("data/test/split/X/" +
                                          self.image_label_directory[fname] +
                                          "/" + fname))

        # print("Shape of X: " + str(X.shape))
        # print("Length of y: " + str(len(y)))

        return X


def main():

    # NOTE: remember, the model used must have been trained with the same tile side 
    #       as the one being used here for testing     

    file_list, dir_list, counts = list_files_from_dir(directory="data/test/split/X",
                                                      extension=".tif")    
    print("\n\nNumber of elements in file list: " + str(len(file_list)))

    ild = {file_list[i]: dir_list[i] for i in range(len(dir_list))}

    model_list, _, _ = list_files_from_dir(directory="models",
                                           extension=".h5")
    print("\nNumber models: " + str(len(model_list)) +"\n\n")

    tile_side = 128
    
    # NOTE: por si solo necesito usar un modelo, puedo usar la linea:
    # model_list = ["20191018_InceptionV3_newPreprocesing_model_35_epochs_15pics_absnorm_x20_128px_full-train.h5"]
    
    for m in model_list:
        print("Model name: " + m)
        model = tf.keras.models.load_model("models/" + m, 
                                        custom_objects={'precision_m': precision_m,
                                                        'recall_m': recall_m,
                                                        'f1_m': f1_m})
        
        test_generator = ImageTestGenerator(list_IDs=file_list,
                                            image_label_directory=ild,
                                            tile_side=tile_side,
                                            batch_size=64)

        results = model.predict_generator(generator=test_generator,
                                        workers=8, 
                                        use_multiprocessing=True,
                                        verbose=1)

        print("shape of results: {}".format(results.shape))
        
        rows = zip(file_list, dir_list, results.reshape(results.shape[0],))
        with open("data/test/split_results/" + m[:-3] + ".csv", "w", newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        K.clear_session()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()





