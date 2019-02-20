import os
from shutil import copyfile, rmtree
import zipfile
import re

dataset = 'potsdam'
images = '2_Ortho_RGB'
labels = '5_Labels_all'

path = os.getcwd()

rmtree(path+'/'+dataset)
os.mkdir(path+'/'+dataset)
os.mkdir(path+'/'+dataset+'/'+images)
os.mkdir(path+'/'+dataset+'/'+labels)

archive = zipfile.ZipFile('../'+dataset+'/'+images+'.zip')
for file in archive.namelist():
    if file.endswith('.tif'):
        archive.extract(file, './'+dataset)

archive = zipfile.ZipFile('../'+dataset+'/'+labels+'.zip')
for file in archive.namelist():
    if file.endswith('.tif'):
        archive.extract(file, './'+dataset+'/'+labels)


id_images = os.listdir('./'+dataset+'/'+images)

for i in range(len(id_images)):
    numbers = re.findall(r'\d+', id_images[i])
    original_name = id_images[i]

    for j in numbers:
        id_images[i] = re.sub('_'+j+'_', '_'+"{:02d}".format(int(j))+'_', id_images[i])
    os.rename('./'+dataset+'/'+images+'/'+original_name, './'+dataset+'/'+images+'/'+id_images[i])

id_images = sorted(id_images)

id_labels = os.listdir('./'+dataset+'/'+labels)

for i in range(len(id_labels)):
    numbers = re.findall(r'\d+', id_labels[i])
    original_name = id_labels[i]
    for j in numbers:
        id_labels[i] = re.sub('_'+j+'_', '_'+"{:02d}".format(int(j))+'_', id_labels[i])
    os.rename('./'+dataset+'/'+labels+'/'+original_name, './'+dataset+'/'+labels+'/'+id_labels[i])
id_labels = sorted(id_labels)

id = 1
for i in id_images:
    os.rename('./'+dataset+'/'+images+'/'+i, './'+dataset+'/'+images+'/'+"{:02d}".format(id)+"_"+i)
    id+=1

id = 1
for i in id_labels:
    os.rename('./'+dataset+'/'+labels+'/'+i, './'+dataset+'/'+labels+'/'+"{:02d}".format(id)+"_"+i)
    id+=1
