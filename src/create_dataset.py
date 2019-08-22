from classes import NDPAnnotationPoint, NDPImage, ImageAnnotationList
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import metrics
import PIL
import os
from time import time
import gc
from skimage import util
from math import floor

'''
    supestos:
    - Para todas las imagenes que se utilizan, el tamaño de los pixeles debe
      ser igual o casi igual (misma resolución)

'''

os.chdir('D:/felipe/ndpi/')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

image_dir = 'D:/felipe/ndpi/'


filename = 'prueba2.ndpi'
annotation_path = "prueba2.ndpi.ndpa"

######### 1. Cargar los instances de imagen y anotacion ############################
# real_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(real_path + "/" + "data/raw")

# filename = "S04_292_p16_RTU_ER1_20 - 2016-04-12 15.42.13.ndpi"
# ndp_image, image_annotation_list = call_ndpi_ndpa(filename)

# image_annotation_list.merge_annotations(draw_mask=True)

########################################
######### 2. Crear imagen y anotacion como numpy ############################

img_name_1 = image_dir + "prueba2_x40_z0_half_1.tif"  # imagen para train
img_name_2 = image_dir + "prueba2_x40_z0_half_2.tif"  # imagen para test

tile_side = 128  # numero de pixeles por lado de cada tile.

result_name = 'results_prueba2_half2_tile{}_v20190509'.format(tile_side) # nombre del resultado, a ser cuardado despues tanto como objeto .npy como imagen .jpeg

# se obtiene la anotacion de intetres
imagen = NDPImage(filename)
annotationList = ImageAnnotationList(ndp_image=imagen,
                                     annotation_path=annotation_path)
annotations = annotationList.annotation_list
annotation_prueba = annotations[0]

# llama la funcion "get_mask" para la anotacion seleccionada
mask = annotation_prueba.get_mask()

##########
# mask_height = round(mask.shape[0] * scale)
# mask_width = round(mask.shape[1] * scale)
# mask = cv2.resize(mask, (mask_width, mask_height), interpolation = cv2.INTER_LINEAR) # image scale
##########

# llama la funcion para guardar una mascara como imagen
# classes.save_np_as_image(mask,'mask.jpeg')
print("Original Parameters:")
print("offset_x, offset_y, mpp_x, mpp_y, width_lvl_0, height_lvl_0")
print(imagen._get_image_properties())

print("mask.shape")
print(mask.shape)

t0 = time()
print("t0: {}".format(t0 % 60))
im1 = cv2.imread(img_name_1)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)

t1 = time() - t0
print("t1: {}".format(t1 % 60))
 
##########
# im1_height = round(im1.shape[0] * scale)
# im1_width = round(im1.shape[1] * scale)
# im1 = cv2.resize(im1, (im1_width, im1_height), interpolation = cv2.INTER_LINEAR) # image scale
##########

im2 = cv2.imread(img_name_2)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

t2 = time() - t1
print("t2: {}".format(t2 % 60))


image_height = im1.shape[0] # 17152 (scale = 1)
image_width = im1.shape[1] # 14336 (scale = 1)
tile_num_expected = floor(image_height/tile_side) * floor(image_width/tile_side) 

print("image height: " + str(image_height))
print("image width: " + str(image_width))
print("expected tile num: " + str(tile_num_expected))




# X = np.zeros((tile_num_expected, tile_side, tile_side, 3)) # se crea un numpy array con las dimensiones esperadas y rellena de 0s,
                                                           # y los 0s se van reemplazando por los valores correctos en el for loop

val_X = np.zeros((tile_num_expected, tile_side, tile_side, 3)) # = []
                                             # por ahora esto sólo va acá porque la imagen es del mismo tamaño que para la prueba
                                             # para generalizar hay que usar los image_height y width de la imagen.


t3 = time() - t2
print("t3: {}".format(t3 % 60))

i=0

n_ver = floor(image_height/tile_side)
n_hor = floor(image_width/tile_side) 
# al restar 1 a image_height/tile_side, nos aseguramos de que si queda un pedazo de la imagen 
# estamos haciendo que no se cuenten los pixeles del borde inferior y derecho, en caso de que
# no alcancen para hacer un tile. E.g. image_height = 10, tile_side = 3. 
# round(image_height/tile_side - 1) arroja 3, por lo que el último pixel no se pesca.
# lo mismo para image width

tile_side_squared = tile_side**2

for ver in range(0, n_ver):

    height_from = tile_side * ver
    height_to = tile_side * (ver + 1)

    for hor in range(0, n_hor):

        width_from = tile_side * hor
        width_to = tile_side * (hor + 1)

        # if np.sum(mask[height_from : height_to, width_from : width_to]) == tile_side_squared:
        #     y[i] = 1
        # else:
        #     y[i] = 0

        # X[i] = im1[height_from : height_to, width_from : width_to, :] / 255
        val_X[i] = im2[height_from : height_to, width_from : width_to, :] / 255
        i += 1

# np.save("D:/felipe/software_projects/epi_seg/src/data/validation/val_X", val_X)
print("valor de i: " + str(i))

tiempo_ = time()

# https://stackoverflow.com/questions/51161999/view-as-windows-on-4d-array

train_img = im1[:image_height - (image_height % tile_side), :image_width - (image_width % tile_side), :]
plt.figure()
plt.imshow(train_img, aspect='auto')
plt.show()

train_img = cv2.normalize(train_img, train_img, 0.,1.,dtype=cv2.CV_32F, norm_type=cv2.NORM_MINMAX)

X = util.view_as_blocks(train_img, block_shape=(tile_side,tile_side,3))
print(X.shape)
X = X.reshape((train_img.shape[0]//tile_side)*(train_img.shape[1]//tile_side), tile_side, tile_side, 3)

print(X.shape)
print("width: " + str(image_width - image_width % tile_side))
print("height: " + str(image_height - image_height % tile_side))


y = [1 
    if np.sum(mask[tile_side * ver : tile_side * (ver + 1),
                   tile_side * hor : tile_side * (hor + 1)
                   ]) == tile_side_squared
    else 0
    for ver in range(0,n_ver)
    for hor in range(0,n_hor)]

unique, counts = np.unique(np.array(y), return_counts=True)

print("mask uniques: " + str(dict(zip(unique, counts))))

results = np.array(y).reshape(n_ver, n_hor)
plt.figure()
plt.imshow(results, aspect='auto')
plt.show()


tiempo_final = round(time()-tiempo_,5)
print("tiempo list comprehenasion: " + str(tiempo_final))

t4 = time() - t3
print("t4: {}".format(t4 % 60))



# print('numero total de tiles segun los utilizado en el loop anterior: ' + str(round(image_height/tile_side - 1) * round(image_width/tile_side - 1)))
# print("numero de tiles positivos: " + str(y.count(1)))
# print("numero de tiles negativos: " + str(y.count(0)))
# print("numero de tiles en el X: " + str(len(X)))

# print("suma total X: " + str(sum(X)))


# print('X.shape: ' + str(X.shape))
# print('X[1].shape: ' + str(X[1].shape))
# print('val_X.shape' + str(val_X.shape))
# print('val_X[1].shape' + str(val_X[1].shape))





# print("marca 1")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=21)
# print("marca 2")




# print(X_train[3000].shape)
# print(X_test[3000].shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                           input_shape=(tile_side, tile_side, 3),
                           data_format="channels_last", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                           data_format="channels_last", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                           data_format="channels_last", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                           data_format="channels_last", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='binary_crossentropy',
            #   loss = 'sparse_categorical_crossentropy',
              metrics=['acc', metrics.Accuracy(), metrics.Precision(), metrics.Recall()])

model.fit(X, y, epochs=5, validation_split=0.5)
# model.save("D:/felipe/software_projects/epi_seg/src/models/create_dataset.h5")

# model.evaluate(X_test, y_test)

results = model.predict_classes(val_X)

t5 = time() - t4
print("t5: {}".format(t5 % 60))

height = n_ver
width =  n_hor



results = results.reshape(height, width)
print('Results - shape: {}'.format(results.shape))
# results = cv2.resize(results, (width*2, height*2),)





plt.figure(figsize = (15, 15))
plt.imshow(results, aspect='auto')
plt.show()

# np.save( result_name + '.npy', results)
# save_np_as_image(results*255, image_dir + result_name + ".jpeg")
