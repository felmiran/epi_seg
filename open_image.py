import cv2
import skimage
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift
# from skimage.morphology import watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from PIL import Image
from time import time

import matplotlib.pyplot as plt
import numpy as np


# img_name = "D:/felipe/ndpi/Untitled.tif"
# img_name = "D:/felipe/ndpi/prueba1_x40_z0_5000x5000.tif"
img_name = "D:/felipe/ndpi/prueba1_x40_z0_10000x10000.tif"
# img_name = "D:/felipe/ndpi/prueba1_x40_z0.tif"
# img_name = "D:/felipe/pictures/foto.jpg"


im = cv2.imread(img_name)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = np.array(im)


# en opencv, al parecer por defecto se lee BGR en vez de RGB, osea las capas están al reves. 
# Im[:,:,0] en verdad es blue y im[:,:,2] es red. 
# Esto se resuelve con el codigo de abajo o con cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# blue = im[:,:,0].copy()
# green = im[:,:,1].copy()
# red = im[:,:,2].copy()

# im[:,:,0] = red
# im[:,:,1] = green
# im[:,:,2] = blue
print(im.shape)

plt.figure(figsize = (15, 15))
plt.imshow(im, aspect='auto')

plt.show()

print("end")