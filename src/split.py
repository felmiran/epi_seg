import os
from classes import NDPImage, ImageAnnotationList
from utils import save_np_as_image, extract_region, build_dirs, \
    list_files_from_dir, normalize_image, call_ndpi_ndpa
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_RGB2GRAY, COLOR_BGR2RGB, \
    normalize, CV_32F, NORM_MINMAX, imread, calcHist, transpose, flip
from math import ceil, floor
import numpy as np
import json
from tqdm import tqdm


def clean_split_files(directory="../split", lista=""):
    '''
    deletes files where mask has no pixel in annotated region
    default directory is "split" from the "raw" folder
    '''
    if lista == "":
        mask_list = os.listdir(directory + "/mask")
        mask_list.remove(".gitkeep")
    else:
        mask_list = [x for x in lista]

    for filename in mask_list:
        mask = imread(directory + "/mask/" + filename)[:, :, 0]

        if np.amax(mask) == 0:
            os.remove(directory + "/mask/" + filename)
            os.remove(directory + "/X/" + filename)


# TODO> registrar para la tesis como se llego a esta funcion tile_is_background
# 1. Se probo con una funcion que calculaba si el % de pixeles entre
#    220 y 240 era mayor a 90% (usando calcHist). Sin embargo, se observo que
#    en la clase 0 tambien habian tiles que eran claramente background,
#    solo que eran un poco mas oscuras.
# 2. Aunque eran mas oscuras, se observaba que la std del histograma era menor
#    en imagenes que son background. Viendo como se distribuye esta std entre
#    tiles (separandolas con el metodo 1) se observo una separacion clara en
#    std=5. Con esa info se hizo la segundafuncion tile_is_background_2.


# TODO> TESTS:
# 1- el nombre de los archivos debe ser igual en X y en mask
# 2- el tamano de los archivos debe ser igual para cada nombre
# 3- los archivos ndpi tienen que tener el formato necesario
# 4- validar que los labels se guarden correctamente, y
#    sobreescriban el archivo antetiot
# 5- test if 5000 por imagen son background. (quizás sea mejor separarlas por
#    carpeta)
# *- (borrar los ?xml y ponerle id a los ndpviewstate)


def rectangle_split_ndpi(ndp_image, split_width, split_height,
                         tohsv=False, path="../split/X"):
    '''
    Splits image into smaller, easier to handle images

    input:
    - ndp_image: object of class NDPImage
    - width:
    - height:
    - lvl:
    - tohsv:

    Observations:
    - Images are saved in the "../split/X" folder with .tif extension

    '''
    width, height = split_width, split_height

    size_hor, size_ver = ndp_image.width_lvl_0, ndp_image.height_lvl_0
    n_hor, n_ver = ceil(size_hor / width), ceil(size_ver / height)

    filename = ndp_image.filename

    print("size horizontal: " + str(ndp_image.width_lvl_0))
    print("size vertical: " + str(ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    lvl = 1

    for h in tqdm(range(n_ver)):
        if h == n_ver-1:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor-1:
                width = size_hor - (n_hor - 1) * split_width

            reg = np.array(ndp_image.read_region(location=(w * width * (lvl+1),
                                                           h * height * (lvl+1)),
                                                 level=lvl,
                                                 size=(width, height))
                           )[:, :, :3]

            dimensions = "_({},{})_{}x{}".format(w*width, h*height,
                                                 width, height)
            split_filename = filename.replace(".ndpi", "") + dimensions

            save_np_as_image(reg, path + "/" + split_filename + ".tif")


def rectangle_split_ndpa(image_annotation_list, split_width,
                         split_height, value_ones=1, path="../split/mask"):

    '''
    mask from image_annotation_list is expected to be 0s and 1s
    '''

    width = split_width
    height = split_height

    merged = image_annotation_list.merge_annotations().mask * value_ones

    size_hor = image_annotation_list.ndp_image.width_lvl_0
    n_hor = ceil(size_hor / width)

    size_ver = image_annotation_list.ndp_image.height_lvl_0
    n_ver = ceil(size_ver / height)

    filename = image_annotation_list.ndp_image.filename

    print("size horizontal: " +
          str(image_annotation_list.ndp_image.width_lvl_0))
    print("size vertical: " +
          str(image_annotation_list.ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    for h in tqdm(range(n_ver)):
        if h == n_ver-1:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor-1:
                width = size_hor - (n_hor - 1) * split_width

            reg = extract_region(merged,
                                 square_top_left_corner=(w*width, h*height),
                                 square_height=height,
                                 square_width=width)
            dimensions = "_({},{})_{}x{}".format(w*width,
                                                 h*height,
                                                 width,
                                                 height)
            filename = filename.replace(".ndpi", "") + dimensions
            save_np_as_image(reg, path + "/" + filename + ".tif")


def main(clean=False):

    '''
    for now, this function grabs a ndpi image, splits the image and the mask
    and saves the splits in the split directory.
    '''
    os.chdir("data/test")

    file_list, _, _ = list_files_from_dir(extension=".ndpi")
    for i, f in enumerate(file_list):
        print("{}. ".format(i) + f)
        
    print(file_list)

    for ndpi_file in file_list:
        print(ndpi_file)
        ndp_image, image_annotation_list = call_ndpi_ndpa(ndpi_file)
        width = floor(ndp_image.width_lvl_0/4)
        height = floor(ndp_image.height_lvl_0/4)
        rectangle_split_ndpi(ndp_image=ndp_image,
                             split_width=width,
                             split_height=height,
                             tohsv=False,
                             path="grandes_RGB")

    if clean:
        clean_split_files()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # build_dirs()

    main(clean=False)
