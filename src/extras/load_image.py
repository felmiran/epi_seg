import cv2
from time import time
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def display_image(image):
    _width = image.shape[1]
    _height = image.shape[0]
    _rounded_ratio = round(_height / _width)
    plt.figure(figsize=(10, 10 * _rounded_ratio))
    plt.imshow(image, aspect='auto')
    plt.show()
    return


# img_name = "D:/felipe/ndpi/Untitled.tif"
# img_name = "D:/felipe/ndpi/prueba1_x40_z0_5000x5000.tif"
# img_name = "D:/felipe/ndpi/prueba1_x40_z0_10000x10000.tif"
img_name_1 = "D:/felipe/ndpi/prueba2_x40_z0_half_1.tif"
img_name_2 = "D:/felipe/ndpi/prueba2_x40_z0_half_2.tif"
# img_name = "D:/felipe/ndpi/prueba1_x40_z0.tif"
# img_name = "D:/felipe/pictures/foto.jpg"


im1 = load_image(img_name_1)

display_image(im1)

# en opencv, al parecer por defecto se lee BGR en vez de RGB, osea las capas
# están al reves.
# Im[:,:,0] en verdad es blue y im[:,:,2] es red.
# Esto se resuelve con el codigo de abajo o con
# cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# blue = im[:,:,0].copy()
# green = im[:,:,1].copy()
# red = im[:,:,2].copy()

# im[:,:,0] = red
# im[:,:,1] = green
# im[:,:,2] = blue


print("end")
