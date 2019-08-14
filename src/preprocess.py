import os
from classes import *
from cv2 import cvtColor, COLOR_RGB2HSV, normalize, CV_32F, NORM_MINMAX
from math import ceil, floor

# TODO> rename to "split.py"


def list_ndpi_files_from_dir():
    # TODO> pasar a "utils.py"
    '''
    lists ndpi files in cwd
    '''
    ndpi_list = os.listdir()
    return [ndpi_file for ndpi_file in ndpi_list
            if ndpi_file.endswith(".ndpi")]


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


# TODO> TESTIS:
# 1- el nombre de los archivos debe ser igual en X y en mask
# 2- el tamano de los archivos debe ser igual para cada nombre
def rectangle_split_ndpi(ndp_image, split_width, split_height, norm=False,
                         tohsv=False, as_numpy=False):
    '''
    Splits image into smaller, easier to handle images

    input:
    - ndp_image: object of class NDPImage
    - width:
    - height:
    - lvl:
    - norm:
    - tohsv:
    - as_numpy:

    Observations:
    - split images can be normalized, but only if these are saved as numpy
      arrays (as_numpy=True); as_numpy=False overrides norm=True.
    - Images are saved in the "../data/split/X" folder with .tif extension

    '''
    width = split_width
    height = split_height

    size_hor = ndp_image.width_lvl_0
    size_ver = ndp_image.height_lvl_0

    n_hor = ceil(size_hor / width)
    n_ver = ceil(size_ver / height)

    print("size horizontal: " + str(ndp_image.width_lvl_0))
    print("size vertical: " + str(ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    lvl = 0
    # original = np.array(ndp_image.read_region(location=(0, 0),
    #                                           level=lvl,
    #                                           size=(size_hor, size_ver)))

    # original = original[:, :, :3]

    # if tohsv:
    #     original = to_hsv(original)

    # if norm and as_numpy:
    #     original = normalize_image(original)

    for h in range(n_ver):
        if h == n_ver:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):
            if w == n_hor:
                width = size_hor - (n_hor - 1) * split_width

            # reg = extract_region(original,
            #                      square_top_left_corner=(w*width, h*height),
            #                      square_height=height,
            #                      square_width=width)

            reg = np.array(ndp_image.read_region(location=(w * width,
                                                           h * height),
                                                 level=lvl,
                                                 size=(width, height)))

            reg = reg[:, :, :3]

            if tohsv:
                reg = to_hsv(reg)

            if norm and as_numpy:
                reg = normalize_image(reg)

            filename = ndp_image.filename
            dimensions = "_({},{})_{}x{}".format(w*width,
                                                 h*height,
                                                 width,
                                                 height)
            filename = filename.replace(".ndpi", "") + dimensions
            if not as_numpy:
                save_mask_as_img(reg, "../split/X/" + filename + ".tif")
            else:
                np.save("../split/X/" + filename, reg)


def rectangle_split_ndpa(image_annotation_list, split_width,
                         split_height, value_ones=1):

    '''
    mask from image_annotation_list is expected to be 0s and 1s
    '''

    width = split_width
    height = split_height

    merged = image_annotation_list.merge_annotations().mask * value_ones

    size_hor = image_annotation_list.ndp_image.width_lvl_0
    size_ver = image_annotation_list.ndp_image.height_lvl_0

    n_hor = ceil(size_hor / width)
    n_ver = ceil(size_ver / height)

    print("size horizontal: " +
          str(image_annotation_list.ndp_image.width_lvl_0))
    print("size vertical: " +
          str(image_annotation_list.ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    for h in range(n_ver):

        if h == n_ver:
            height = size_ver - (n_ver - 1) * split_height

        for w in range(n_hor):

            if w == n_hor:
                width = size_hor - (n_hor - 1) * split_width

            reg = extract_region(merged,
                                 square_top_left_corner=(w*width, h*height),
                                 square_height=height,
                                 square_width=width)
            filename = image_annotation_list.ndp_image.filename
            dimensions = "_({},{})_{}x{}".format(w*width,
                                                 h*height,
                                                 width,
                                                 height)
            filename = filename.replace(".ndpi", "") + dimensions
            save_mask_as_img(reg, "../split/mask/" + filename + ".tif")


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


def clean_empty_splits():
    # TODO>
    '''
    deletes image and mask splits where there is no annotated tile
    '''
    pass


def main():

    '''
    for now, this function grabs a ndpi image, splits the image and the mask
    and saves the splits in the split directory

    '''
    os.chdir("data/raw")
    for ndpi_file in list_ndpi_files_from_dir():
        print(ndpi_file)
        ndp_image, image_annotation_list = call_ndpi_ndpa(ndpi_file)
        width, height = 128, floor(ndp_image.height_lvl_0/2)
        rectangle_split_ndpa(image_annotation_list=image_annotation_list,
                             split_width=width,
                             split_height=height,
                             value_ones=1)
        rectangle_split_ndpi(ndp_image=ndp_image,
                             split_width=width,
                             split_height=height,
                             norm=True,
                             tohsv=True,
                             as_numpy=False)

    # print(os.getcwd())  # borrar
    # os.chdir("data/raw")
    # filename = "S04_292_p16_RTU_ER1_20 - 2016-04-12 15.42.13.ndpi"
    # ndp_image, image_annotation_list = call_ndpi_ndpa(filename)
    # width, height = 128, floor(ndp_image.height_lvl_0/2)
    # rectangle_split_ndpa(image_annotation_list=image_annotation_list,
    #                      width=width,
    #                      height=height,
    #                      value_ones=1)
    # rectangle_split_ndpi(ndp_image=ndp_image,
    #                      width=width,
    #                      height=height,
    #                      norm=True,
    #                      tohsv=True,
    #                      as_numpy=False)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
