import os
from classes import NDPImage, ImageAnnotationList
from utils import save_np_as_image, extract_region, build_dirs, \
    list_files_from_dir, to_hsv, normalize_image, call_ndpi_ndpa
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_RGB2GRAY, COLOR_BGR2RGB, \
    NORM_MINMAX, imread, calcHist, transpose, flip
from math import ceil, floor
import numpy as np
import json
from tqdm import tqdm


def data_augmentation(directory="../split/X/1", flip_imgs=True):
    '''
    generates extra images for eah image in directory: 3 addicional
    images corresponding to 90, 180, 270 rotations.
    if flip=True the flipped version of the 4 images (original + 3)
    will be created as well.
    '''
    labels = {}

    print("Data augmentation commencing for images labelled 'epithelium'.")
    print("flip_imgs={}".format(flip_imgs))

    file_list, _ = list_files_from_dir(directory=directory, extension=".tif")
    for img_name in tqdm(file_list):

        img = imread(directory + "/" + img_name)

        img_90, labels["90_" + img_name] = flip(transpose(img), 1), 1
        save_np_as_image(cvtColor(img_90, COLOR_BGR2RGB),
                         directory + "/90_" + img_name)

        img_180, labels["180_" + img_name] = flip(img, -1), 1
        save_np_as_image(cvtColor(img_180, COLOR_BGR2RGB),
                         directory + "/180_" + img_name)

        img_270, labels["270_" + img_name] = flip(transpose(img), 0), 1
        save_np_as_image(cvtColor(img_270, COLOR_BGR2RGB),
                         directory + "/270_" + img_name)

        if flip_imgs:
            img_f, labels["f_" + img_name] = flip(img, 1), 1
            save_np_as_image(cvtColor(img_f, COLOR_BGR2RGB),
                             directory + "/f_" + img_name)

            img_90f, labels["90f_" + img_name] = flip(img_90, 1), 1
            save_np_as_image(cvtColor(img_90f, COLOR_BGR2RGB),
                             directory + "/90f_" + img_name)

            img_180f, labels["180f_" + img_name] = flip(img_180, 1), 1
            save_np_as_image(cvtColor(img_180f, COLOR_BGR2RGB),
                             directory + "/180f_" + img_name)

            img_270f, labels["270f_" + img_name] = flip(img_270, 1), 1
            save_np_as_image(cvtColor(img_270f, COLOR_BGR2RGB),
                             directory + "/270f_" + img_name)

    json.dump(labels, open("../split/X/augmentations.txt", "w"))

    pass


# TODO> registrar para la tesis como se llego a esta funcion tile_is_background
# 1. Se probo con una funcion que calculaba si el % de pixeles entre
#    220 y 240 era mayor a 90% (usando calcHist). Sin embargo, se observo que
#    en la clase 0 tambien habian tiles que eran claramente background,
#    solo que eran un poco mas oscuras.
# 2. Aunque eran mas oscuras, se observaba que la std del histograma era menor
#    en imagenes que son background. Viendo como se distribuye esta std entre
#    tiles (separandolas con el metodo 1) se observo una separacion clara en
#    std=5. Con esa info se hizo la segundafuncion tile_is_background_2.


def tile_is_background_2(image, threshold=5):
    '''
    returns True if tile (np array) is background. An <image> is classified
    as background if std dev of pixel colors (gray scale, 0-255) is over
    <threshold>.
    '''
    is_bkgnd = False
    std = np.std(cvtColor(image, COLOR_RGB2GRAY))
    if std < threshold:
        is_bkgnd = True
    return std, is_bkgnd


def tile_is_background_1(image, rng=(220, 240), threshold=0.9):
    '''
    returns True if tile (np array) is background. An <image> is classified
    as background if the proportion of pixels colors (gray scale, 0-255)
    within <rng> is over <threshold>.

    inputs:
     - image: numpy array corresponding to RGB image
     - rng: range of values to evaluate in histogram
     - threshold: if (pixels in rng / total pixels)
                is higher than threshold, images is classified as background
    '''
    is_bkgnd = False
    hist = calcHist(images=[cvtColor(image, COLOR_RGB2GRAY)],
                    channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    if np.sum(hist[rng[0]:rng[1]])/np.sum(hist) > 0.9:
        is_bkgnd = True
    return hist, is_bkgnd


# TODO> TESTS:
# 1- el nombre de los archivos debe ser igual en X y en mask
# 2- el tamano de los archivos debe ser igual para cada nombre
# 3- los archivos ndpi tienen que tener el formato necesario
# 4- validar que los labels se guarden correctamente, y
#    sobreescriban el archivo antetiot
# 5- test if 5000 por imagen son background. (quizás sea mejor separarlas por
#    carpeta)
# *- (borrar los ?xml y ponerle id a los ndpviewstate)


def rectangle_split_ndpi_ndpa(ndp_image, image_annotation_list, split_height,
                              split_width, tohsv=False, path_ndpi="../split/X",
                              path_ndpa="../split/mask", n_bkgnd_tiles=5000):
    '''
    Splits ndpi into tiles and saves lanel as dict in "labels.txt".
    Keys are filenames.
    '''
    merged = image_annotation_list.merge_annotations().mask

    width, height = split_width, split_height
    size_hor, size_ver = ndp_image.width_lvl_0, ndp_image.height_lvl_0
    n_hor, n_ver = ceil(size_hor / width), ceil(size_ver / height)

    filename = ndp_image.filename
    print("filename: " + filename)

    print("size horizontal: " + str(ndp_image.width_lvl_0))
    print("n of horizontal splits: " + str(n_hor))
    print("size vertical: " + str(ndp_image.height_lvl_0))
    print("n of vertical splits: " + str(n_ver))

    labels = {}
    bkgnd_tiles_counter = 0
    tile_class = "0"

    for h in tqdm(range(n_ver)):
        if h == n_ver-1:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor-1:
                width = size_hor - (n_hor - 1) * split_width

            reg_ndpi = np.array(ndp_image.read_region(location=(w * width,
                                                                h * height),
                                                      level=0, size=(width,
                                                                     height)
                                                      ))[:, :, :3]

            reg_ndpa = extract_region(merged,
                                      square_top_left_corner=(w * width,
                                                              h * height),
                                      square_height=height, square_width=width)

            dimensions = "_({},{})_{}x{}".format(w * width, h * height, width,
                                                 height)
            split_filename = filename.replace(".ndpi", "")
            split_filename = split_filename + dimensions + ".tif"

            # _, is_bkgnd = tile_is_background_1(reg_ndpi, rng=(220, 240),
            #                                    threshold=0.9)

            _, is_bkgnd = tile_is_background_2(reg_ndpi, threshold=5)

            if is_bkgnd:
                if bkgnd_tiles_counter < n_bkgnd_tiles:
                    labels[split_filename] = 2
                    bkgnd_tiles_counter += 1
                    tile_class = "-1"
                else:
                    continue
            else:
                if np.sum(reg_ndpa) == height * width:
                    labels[split_filename] = 1
                    tile_class = "1"
                else:
                    labels[split_filename] = 0
                    tile_class = "0"

            if tohsv:
                reg_ndpi = to_hsv(reg_ndpi)

            save_np_as_image(reg_ndpi, path_ndpi + "/" + tile_class +
                             "/" + split_filename)

    json.dump(labels, open(path_ndpi + "/" + filename + ".txt", "w"))


def main():

    '''
    for now, this function grabs a ndpi image, splits the image and the mask
    and saves the splits in the split directory.
    '''
    os.chdir("data/raw")

    file_list, _ = list_files_from_dir(extension=".ndpi")
    print(file_list)
    width, height = 128, 128
    print("Tile size: {}x{}".format(height, width))

    for ndpi_file in file_list:
        print(ndpi_file)
        rectangle_split_ndpi_ndpa(ndp_image=ndp_image,
                                  image_annotation_list=image_annotation_list,
                                  split_height=height,
                                  split_width=width,
                                  tohsv=False,
                                  path_ndpi="../split/X",
                                  path_ndpa="../split/mask")

    data_augmentation()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # build_dirs()

    main()
