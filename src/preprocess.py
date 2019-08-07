from classes import *
from cv2 import *
from math import ceil


def call_ndpi_ndpa(filename):
    '''
    Filename is the name of the ndpi file.
    OpenSlide requires the file to be in the cwd.

    '''
    ndp_image = NDPImage(filename)
    image_annotation_list = ImageAnnotationList(ndp_image, filename + ".ndpa")
    return ndp_image, image_annotation_list


# TODO
def rectangle_split_ndpi(ndp_image, width, height, lvl=0, norm=False, tohsv=False, as_numpy=False):
    '''
    splits image into smaller, easier to handle images

    '''
    size_hor = ndp_image.width_lvl_0
    size_ver = ndp_image.height_lvl_0

    n_hor = ceil(size_hor / width)
    n_ver = ceil(size_ver / height)

    print("size horizontal: " + str(ndp_image.width_lvl_0))
    print("size vertical: " + str(ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    original = np.array(ndp_image.read_region(location=(0, 0),
                                              level=lvl,
                                              size=(size_hor, size_ver)))

    original = original[:, :, :3]
    if tohsv:
        original = to_hsv(original)

    if norm and as_numpy:
        original = normalize_image(original)

    for h in range(n_ver):
        for w in range(n_hor):
            reg = extract_region(original,
                                 square_top_left_corner=(w*width, h*height),
                                 square_height=height,
                                 square_width=width)
            filename = ndp_image.filename
            dimensions = "({},{})_{}x{}.tif".format(w*width,
                                                    h*height,
                                                    width,
                                                    height)
            filename = filename.replace(".ndpi", "") + dimensions

            if not as_numpy:
                save_mask_as_img(reg, "../split/X/" + filename + ".tif")
            else:
                np.save("../split/X/" + filename, reg)


def rectangle_split_ndpa(image_annotation_list, width, height, value_ones=1):

    '''
    mask from image_annotation_list is expected to be 0s and 1s
    '''

    merged = image_annotation_list.merge_annotations().mask * value_ones

    size_hor = image_annotation_list.ndp_image.width_lvl_0
    size_ver = image_annotation_list.ndp_image.height_lvl_0

    n_hor = ceil(size_hor / width)
    n_ver = ceil(size_ver / height)

    print("size horizontal: " + str(image_annotation_list.ndp_image.width_lvl_0))
    print("size vertical: " + str(image_annotation_list.ndp_image.height_lvl_0))

    print("n_hor: " + str(n_hor))
    print("n_ver: " + str(n_ver))

    for h in range(n_ver):
        for w in range(n_hor):
            reg = extract_region(merged,
                                 square_top_left_corner=(w*width, h*height),
                                 square_height=height,
                                 square_width=width)
            filename = image_annotation_list.ndp_image.filename
            dimensions = "({},{})_{}x{}".format(w*width,
                                                h*height,
                                                width,
                                                height)
            filename = filename.replace(".ndpi", "") + dimensions
            save_mask_as_img(reg, "../split/mask/" + filename + ".tif")


def to_hsv(image):
    '''
    Image corresponds to a numpy array.
    function equires original image to come in BGR.
    NDPI images come like this.
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def normalize_image(image):
    '''
    image corresponds to a numpy array.
    '''

    return cv2.normalize(image, None, alpha=0., beta=1.,
                         dtype=cv2.CV_32F, norm_type=cv2.NORM_MINMAX)


def main():
    print(os.getcwd())  # borrar
    os.chdir("data/raw")
    filename = "S04_292_p16_RTU_ER1_20 - 2016-04-12 15.42.13.ndpi"
    ndp_image, image_annotation_list = call_ndpi_ndpa(filename)
    width, height = 1280*4, 1280*4
    # rectangle_split_ndpa(image_annotation_list=image_annotation_list,
    #                      width=width,
    #                      height=height,
    #                      value_ones=255)
    rectangle_split_ndpi(ndp_image=ndp_image,
                         width=width,
                         height=height,
                         norm=True,
                         tohsv=True,
                         as_numpy=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    main()
