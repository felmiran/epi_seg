import classes
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import PIL
import os

def save_mask_as_img(numpy_array, filename):
    im = PIL.Image.fromarray(np.uint8(numpy_array*255))
    im.save(filename)
    return



'''
    supestos:
    - Para todas las imagenes que se utilizan, el tamaño de los pixeles debe ser igual o casi igual (misma resolución)

'''



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

image_dir = 'D:/felipe/ndpi/'


image_path = image_dir + 'prueba2.ndpi' 
annotation_path = image_dir + "prueba2.ndpi.ndpa"

img_name_1 = image_dir + "prueba2_x40_z0_half_1.tif" # imagen para entrenamiento
img_name_2 = image_dir + "prueba2_x40_z0_half_2.tif" # imagen para validacion

tile_side = 64  # numero de pixeles por lado de cada tile. En este caso solo es un cuadrado.

scale = 1 # Valor entre 1 y 0. Si no se quiere achicar la imagen, se deja como 1. Si se quiere rebajar en un 50% 

result_name = 'results_prueba2_half2_tile128_v4' # nombre del resultado, a ser cuardado despues tanto como objeto .npy como imagen .jpeg



# se obtiene la anotacion de intetres
imagen = classes.ImageProperties(image_path)
annotationList = classes.ImageAnnotationList(associated_image=imagen, annotation_path=annotation_path)
annotations = annotationList.annotation_list
annotation_prueba = annotations[0]

# llama la funcion "get_mask" para la anotacion seleccionada
mask = annotation_prueba.get_mask()

##########
mask_height = round(mask.shape[0] * scale)
mask_width = round(mask.shape[1] * scale)
mask = cv2.resize(mask, (mask_width, mask_height), interpolation = cv2.INTER_LINEAR) # image scale
##########

# llama la funcion para guardar una mascara como imagen
# classes.save_mask_as_img(mask,'mask.jpeg')
print("Original Parameters:")
print("offset_x, offset_y, mpp_x, mpp_y, width_lvl_0, height_lvl_0")
print(imagen._get_image_parameters())

print("mask.shape")
print(mask.shape)

im1 = cv2.imread(img_name_1)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
 
##########
im1_height = round(im1.shape[0] * scale)
im1_width = round(im1.shape[1] * scale)
im1 = cv2.resize(im1, (im1_width, im1_height), interpolation = cv2.INTER_LINEAR) # image scale
##########

im2 = cv2.imread(img_name_2)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

##########
im2_height = round(im2.shape[0] * scale)
im2_width = round(im2.shape[1] * scale)
im2 = cv2.resize(im2, (im2_width, im2_height), interpolation = cv2.INTER_LINEAR) # image scale
##########

image_height = im1.shape[0] # 17152 (scale = 1)
image_width = im1.shape[1] # 14336 (scale = 1)
tile_num_expected = round(image_height/tile_side - 1) * round(image_width/tile_side - 1) # 272367 (scale = 1 y tile_side = 30)

print("image height: " + str(image_height))
print("image width: " + str(image_width))
print("expected tile num: " + str(tile_num_expected))



# Idea: en vez de hacer estas listas y despues pasarlas a numpy array, crear un numpy array con las dimensiones esperadas
#       e ir rellenando en el "for"


X = np.zeros((tile_num_expected, tile_side, tile_side, 3)) # = []
y = []


validation_X = np.zeros((tile_num_expected, tile_side, tile_side, 3)) # = []
                                             # por ahora esto sólo va acá porque la imagen es del mismo tamaño que para la prueba
                                             # para generalizar hay que usar los image_height y width de la imagen.


i=0
for ver in range(0,round(image_height/tile_side - 1)):
    for hor in range(0,round(image_width/tile_side - 1)):
        if np.sum(mask[tile_side * ver : tile_side * (ver + 1), tile_side * hor : tile_side * (hor + 1)]) == tile_side**2:
            y.append(1)
        else:
            y.append(0)
        
        '''
        # X.append(im1[tile_side * ver : tile_side * (ver + 1), tile_side * hor : tile_side * (hor + 1),:] / 255)
        # validation_X.append(im2[tile_side * ver : tile_side * (ver + 1), tile_side * hor : tile_side * (hor + 1),:] / 255)
        '''
        
        X[i] = im1[tile_side * ver : tile_side * (ver + 1), tile_side * hor : tile_side * (hor + 1),:] / 255
        validation_X[i] = im2[tile_side * ver : tile_side * (ver + 1), tile_side * hor : tile_side * (hor + 1),:] / 255
        i+=1


print("valor de i: " + str(i))
# print('numero total de tiles segun los utilizado en el loop anterior: ' + str(round(image_height/tile_side - 1) * round(image_width/tile_side - 1)))
# print("numero de tiles positivos: " + str(y.count(1)))
# print("numero de tiles negativos: " + str(y.count(0)))
# print("numero de tiles en el X: " + str(len(X)))

# print("suma total X: " + str(sum(X)))

'''
# X = np.array(X)
# validation_X = np.array(validation_X)
'''


print('X.shape: ' + str(X.shape))
print('X[1].shape: ' + str(X[1].shape))
print('validation_X.shape' + str(validation_X.shape))
print('validation_X[1].shape' + str(validation_X[1].shape))

# image height: 17152
# image width: 14336
# X.shape: (272367, 30, 30, 3)
# X[1].shape: (30, 30, 3)
# validation_X.shape(272367, 30, 30, 3)
# validation_X.shape(272367, 30, 30, 3)

print("marca 1")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=21)
print("marca 2")


# print(X_train[3000].shape)
# print(X_test[3000].shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), input_shape=(tile_side, tile_side, 3), data_format="channels_last", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), data_format="channels_last", activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
	# tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(256, activation=tf.nn.relu),
	# tf.keras.layers.Dropout(0.2),
	# tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
            #   loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test)

results = model.predict_classes(validation_X)
results = results.reshape(round(image_height / tile_side - 1), round(image_width / tile_side - 1))

plt.figure(figsize = (15, 15))
plt.imshow(results, aspect='auto')

plt.show()

# np.save( result_name + '.npy', results)
# save_mask_as_img(results, image_dir + result_name + ".jpeg")



