import os
from classes import *
from cv2 import cvtColor, COLOR_RGB2HSV, COLOR_RGB2GRAY, normalize, CV_32F, NORM_MINMAX, imread, calcHist
from math import ceil, floor
import numpy as np
import json

# TODO> rename to "split.py"


def list_files_from_dir(directory=None, extension=".ndpi"):
    # TODO> pasar a "utils.py"
    '''
    lists ndpi files in cwd
    '''
    ndpi_list = os.listdir(directory)
    return [ndpi_file for ndpi_file in ndpi_list
            if ndpi_file.endswith(extension)]


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


def call_ndpi_ndpa(filename):
    # TODO pasar a "utils.py"
    '''
    Filename is the name of the ndpi file.
    OpenSlide requires the file to be in the cwd.
    the annotation must end in ".ndpi.ndpa".
    '''
    ndp_image = NDPImage(filename)
    image_annotation_list = ImageAnnotationList(ndp_image, filename + ".ndpa")
    return ndp_image, image_annotation_list


def tile_is_background(image, rng=(220, 240), threshold=0.9):
    '''
    returns True if tile (np array) is not background. An <image> is classified
    as background if the proportion of pixels within <rng> is over
    <threshold>.

    inputs:
     - image: numpy array corresponding to RGB image
     - rng: range of values to evaluate in histogram
     - threshold: % over which tile is classified as bacground
    '''
    is_bkgnd = False
    hist = calcHist(images=[cvtColor(image, COLOR_RGB2GRAY)],
                    channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    if np.sum(hist[rng[0]:rng[1]])/np.sum(hist) > 0.9:
        is_bkgnd = True
    return is_bkgnd


# TODO> TESTS:
# 1- el nombre de los archivos debe ser igual en X y en mask
# 2- el tamano de los archivos debe ser igual para cada nombre
# 3- los archivos ndpi tienen que tener el formato necesario
#    (borrar los ?xml y ponerle id a los ndpviewstate)


def rectangle_split_ndpi_ndpa(ndp_image, image_annotation_list, split_height,
                              split_width, tohsv=False, path_ndpi="../split/X",
                              path_ndpa="../split/mask"):
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

    for h in range(n_ver):
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

            if np.sum(reg_ndpa) == height * width:
                is_background = tile_is_background(reg_ndpi, rng=(220, 240),
                                                   threshold=0.9)
                if is_background:
                    continue
                labels[split_filename] = 1
            else:
                continue  # borrar despues de esta prueba
                labels[split_filename] = 0

            if tohsv:
                reg_ndpi = to_hsv(reg_ndpi)

            save_np_as_image(reg_ndpi, path_ndpi + "/" + split_filename)

    json.dump(labels, open(path_ndpi + "/" + filename + ".txt", "w"))



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
    - Images are saved in the "../data/split/X" folder with .tif extension

    '''
    width = split_width
    height = split_height

    size_hor = ndp_image.width_lvl_0
    n_hor = ceil(size_hor / width)

    size_ver = ndp_image.height_lvl_0
    n_ver = ceil(size_ver / height)

    filename = ndp_image.filename

    print("size horizontal: " + str(ndp_image.width_lvl_0))
    print("size vertical: " + str(ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    lvl = 0

    for h in range(n_ver):
        if h == n_ver-1:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor-1:
                width = size_hor - (n_hor - 1) * split_width

            reg = np.array(ndp_image.read_region(location=(w * width,
                                                           h * height),
                                                 level=lvl,
                                                 size=(width, height))
                           )[:, :, :3]

            if tohsv:
                reg = to_hsv(reg)

            dimensions = "_({},{})_{}x{}".format(w*width, h*height,
                                                 width, height)
            filename = filename.replace(".ndpi", "") + dimensions
            
            save_np_as_image(reg, path + "/" + filename + ".tif")


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

    for h in range(n_ver):
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



def to_hsv(image):
    # TODO pasar a "utils.py"
    '''
    Image corresponds to a numpy array.
    function equires original image to come in RGB.
    NDPI images extracted with Openslide come in RGBA,
    so the last channel is dismissed.
    '''
    return cvtColor(image, COLOR_RGB2HSV)


def normalize_image(image):
    # TODO pasar a "utils.py"
    '''
    image corresponds to a numpy array.
    '''

    return normalize(image, None, alpha=0., beta=1.,
                     dtype=CV_32F, norm_type=NORM_MINMAX)


def main(clean=False):

    '''
    for now, this function grabs a ndpi image, splits the image and the mask
    and saves the splits in the split directory.
    '''
    os.chdir("data/raw")

    archivo = ["S04_3441_p16_RTU_ER1_20 - 2016-04-12 15.45.38.ndpi"]
    for ndpi_file in archivo:
    # for ndpi_file in list_files_from_dir():
        print(ndpi_file)
        ndp_image, image_annotation_list = call_ndpi_ndpa(ndpi_file)
        # width, height = floor(ndp_image.width_lvl_0/4), floor(ndp_image.height_lvl_0/4)
        # width, height = 128, 9600
        # rectangle_split_ndpa(image_annotation_list=image_annotation_list,
        #                      split_width=width,
        #                      split_height=height,
        #                      value_ones=1)
        # rectangle_split_ndpi(ndp_image=ndp_image,
        #                      split_width=width,
        #                      split_height=height,
        #                      tohsv=False)

        # width, height = 9600, 128
        # rectangle_split_ndpa(image_annotation_list=image_annotation_list,
        #                      split_width=width,
        #                      split_height=height,
        #                      value_ones=1)
        # rectangle_split_ndpi(ndp_image=ndp_image,
        #                      split_width=width,
        #                      split_height=height,
        #                      tohsv=True)
        width, height = 128, 128
        rectangle_split_ndpi_ndpa(ndp_image=ndp_image,
                                           image_annotation_list=image_annotation_list,
                                           split_height=height,
                                           split_width=width,
                                           tohsv=False,
                                           path_ndpi="../split/X",
                                           path_ndpa="../split/mask")
                
        

    # if clean:
    #     clean_split_files()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    files = os.listdir("data/split/X")
    for f in files:
        if f.endswith(".tif"):
            try:
                os.remove("data/split/X/" + f)
            except:
                pass
            try:
                os.remove("data/split/mask/" + f)
            except:
                pass
            
    main(clean=False)
