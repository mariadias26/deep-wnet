import rasterio
import tifffile as tiff
import numpy as np
import glob
import os


mypath='./potsdam/5_Labels_all/*.tif'
mylist = [os.path.basename(f) for f in glob.glob(mypath)]
path_dest='./potsdam/5_Labels_all_norm/*.tif'
dest =  [os.path.basename(f) for f in glob.glob(path_dest)]

dict_all = dict()

    #img_m = normalize(tiff.imread(name_img).transpose([1, 2, 0]))
for  filename in mylist:
    print("----",filename)
    if filename in dest:
        print('done')
        continue
    if "4_12" not in filename:
        continue
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

            if key not in dict_all:
                dict_all[key] = k
                k+=1
            if key not in dict_classes:
                dict_classes[key] = np.zeros(np.shape(r))
                dict_classes_color[(r[i,j],g[i,j],b[i,j])] = [i,j,0]
            #dict_classes_color[(r[i,j],g[i,j],b[i,j])][2] +=1
            dict_classes[key][i,j] = 1
    aux =[0] * len(dict_all.values())
    print(dict_classes.keys())
    for key in dict_all.keys():
        if key not in dict_classes:
            dict_classes[key] = np.zeros(np.shape(r))
        aux[dict_all[key]] = dict_classes[key]
    print("colors ",dict_classes_color)

    mask = np.stack(tuple(aux),axis=-1)
    print('Image ', filename, ' read')
    print(np.shape(mask))
    if not os.path.exists('./potsdam/5_Labels_all_norm'):
        os.makedirs('./potsdam/5_Labels_all_norm')
    tiff.imsave('./potsdam/5_Labels_all_norm/{}'.format(filename), mask)
