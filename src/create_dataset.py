from classes import *
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
# classes.save_mask_as_img(mask,'mask.jpeg')
print("Original Parameters:")
print("offset_x, offset_y, mpp_x, mpp_y, width_lvl_0, height_lvl_0")
print(imagen._get_image_properties())

print("mask.shape")
print(mask.shape)

t0 = time()
print("t0: {}".format(t0 % 60))
im1 = cv2.imread(img_name_1)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)

t1 = t0 - time()
print("t1: {}".format(t1 % 60))
 
##########
# im1_height = round(im1.shape[0] * scale)
# im1_width = round(im1.shape[1] * scale)
# im1 = cv2.resize(im1, (im1_width, im1_height), interpolation = cv2.INTER_LINEAR) # image scale
##########

im2 = cv2.imread(img_name_2)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

t2 = t1 - time()
print("t2: {}".format(t2 % 60))

##########
# im2_height = round(im2.shape[0] * scale)
# im2_width = round(im2.shape[1] * scale)
# im2 = cv2.resize(im2, (im2_width, im2_height), interpolation = cv2.INTER_LINEAR) # image scale
##########

image_height = im1.shape[0] # 17152 (scale = 1)
image_width = im1.shape[1] # 14336 (scale = 1)
tile_num_expected = round(image_height/tile_side - 1) * round(image_width/tile_side - 1) # 272367 (scale = 1 y tile_side = 30)

print("image height: " + str(image_height))
print("image width: " + str(image_width))
print("expected tile num: " + str(tile_num_expected))




X = np.zeros((tile_num_expected, tile_side, tile_side, 3)) # se crea un numpy array con las dimensiones esperadas y rellena de 0s,
                                                           # y los 0s se van reemplazando por los valores correctos en el for loop

val_X = np.zeros((tile_num_expected, tile_side, tile_side, 3)) # = []
                                             # por ahora esto sólo va acá porque la imagen es del mismo tamaño que para la prueba
                                             # para generalizar hay que usar los image_height y width de la imagen.


t3 = t2 - time()
print("t3: {}".format(t3 % 60))

i=0

n_ver = round(image_height/tile_side - 1)
n_hor = round(image_width/tile_side - 1)
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

        X[i] = im1[height_from : height_to, width_from : width_to, :] / 255
        val_X[i] = im2[height_from : height_to, width_from : width_to, :] / 255
        i += 1

        # gc.collect() # esta funcion es como un 'empty recycle bin' donde se borran los unreferenced objects

tiempo_ = time()
y = [
    1 
    if np.sum(mask[tile_side * ver : tile_side * (ver + 1), tile_side * hor : tile_side * (hor + 1)]) == tile_side_squared
    else 0
    for hor in range(0,n_hor)
    for ver in range(0,n_ver)
    ]


tiempo_final = round(time()-tiempo_,5)
print("tiempo list comprehenasion: " + str(tiempo_final))

t4 = t3 - time()
print("t4: {}".format(t4 % 60))


print("valor de i: " + str(i))
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

# model.evaluate(X_test, y_test)

results= model.predict_classes(val_X)

t5 = t4 - time()
print("t5: {}".format(t5 % 60))




height = round(image_height / tile_side - 1)
width =  round(image_width / tile_side - 1)



results = results.reshape(height, width)
print('Results - shape: {}'.format(results.shape))
# results = cv2.resize(results, (width*2, height*2),)





plt.figure(figsize = (15, 15))
plt.imshow(results, aspect='auto')

plt.show()

# np.save( result_name + '.npy', results)
save_mask_as_img(results*255, image_dir + result_name + ".jpeg")



