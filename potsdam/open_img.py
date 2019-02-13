import zipfile
from os import listdir
from os.path import isfile, join

labels = './5_Labels_all'
ground_truth = [f for f in listdir(labels) if isfile(join(labels, f))]
print(len(ground_truth))
archive = zipfile.ZipFile('2_Ortho_RGB.zip')

for file in archive.namelist():
    if file.endswith('.tif'):
        archive.extract(file, 'destination_path')
