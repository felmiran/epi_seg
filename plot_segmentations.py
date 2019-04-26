import matplotlib.pyplot as plt
import numpy as np

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



# img_name = "D:/felipe/ndpi/prueba1_x40_z0_250x250.tif"
img_name = "D:/felipe/ndpi/prueba1_x40_z0_1000x1000.tif"


img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = np.array(img)


t0 = time()
print("started")
segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)

t1 = time()
print ('training time: ' + str(round(t1-t0,3)))


segments_slic = slic(img, n_segments=1000, compactness=10, sigma=1)

t2 = time()
print ('training time: ' + str(round(t2-t1,3)))

segments_quick = quickshift(img, kernel_size=5, max_dist=6, ratio=0.5)

t3 = time()
print ('training time: ' + str(round(t3-t2,3)))



print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

t4 = time()
print ('training time: ' + str(round(t4-t3,3)))

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(img)
ax[1, 1].set_title('Original')

for a in ax.ravel():
    a.set_axis_off()

# plt.tight_layout()
# plt.show()


print ("----------------------------------------")
print(segments_fz)