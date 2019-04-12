from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction import image
from keras.preprocessing.image import img_to_array, array_to_img
import tifffile as tiff
import rasterio
import numpy as np

path = 'potsdam/mask'
path_img = './potsdam/mask/mask/top_potsdam_{}_label.tif'

train_id = ['3_10','3_11','3_12']
'''
for id in train_id:
    img = tiff.imread(path_img.format(id))
    #tiff.imsave(path_img.format(id), img)
'''
image_datagen = ImageDataGenerator()

image_generator = image_datagen.flow_from_directory(path,batch_size=5,class_mode=None)

print(np.shape(image_generator.next()))
