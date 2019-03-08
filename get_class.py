import os
import rasterio
import tifffile as tiff
import numpy as np
import glob

mypath='./potsdam/5_Labels_all/*.tif'
mylist = [os.path.basename(f) for f in glob.glob(mypath)]
path_dest='./potsdam/5_Labels_all_norm/*.tif'
dest =  [os.path.basename(f) for f in glob.glob(path_dest)]
filename = dest[2]
print(filename)
old_mask = tiff.imread('./potsdam/5_Labels_all_norm/{}'.format(filename))

dict_all = dict()

mask = tiff.imread('./potsdam/5_Labels_all/{}'.format(filename))
print(np.shape(mask))
r = mask[:,:,0]
g = mask[:,:,1]
b = mask[:,:,2]

new = np.empty(np.shape(r))
x, y = np.shape(new)
dict_classes = dict()
dict_classes_color = dict()

k = 0
for i in range(0, x):
    for j in range(0,y):
        key = (round(int(r[i,j])/255,0),round(int(g[i,j])/255,0),round(int(b[i,j])/255,0))
        if key not in dict_classes:
            for stuff in range(0,6):
                if old_mask[i,j,stuff] == 1.0:
                    dict_classes_color[(r[i,j],g[i,j],b[i,j])] = stuff
        #dict_classes_color[(r[i,j],g[i,j],b[i,j])][2] +=1
print(dict_classes_color)
