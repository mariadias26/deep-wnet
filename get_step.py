import tifffile as tiff
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def find_step(img):
  x_original, y_original, ch = img.shape
  img_x, img_y, ch = img.shape
  to_iterate = 'x'
  while True:
    for i in range(10, 25):
      a = (img_x - patch_sz)/i
      b = (img_y - patch_sz)/i
      if int(a) == a and int(b) == b:
        return i, img_x, img_y, x_original, y_original
      elif int(a) == a:
        y = img_y
        while (y - img_y) < 10 :
          b = (y - patch_sz)/i
          if int(b) == b:
            return i, img_x, y, x_original, y_original
          else:
            y +=1
      elif int(b) == b:
        x  = img_x
        while (x - img_x) < 10:
          a = (x - patch_sz)/i
          if int(a) == a:
            return i, x, img_y, x_original, y_original
          else:
            x +=1
    if to_iterate == 'x':
      img_x+=1
      to_iterate = 'y'
    else:
      img_y+=1
      to_iterate = 'x'



#img = cv2.copyMakeBorder( img_original, 0, padding_x, 0, padding_y, cv2.BORDER_CONSTANT)
#rec = img[:dim_x, :dim_y, ]
