from os import listdir
from os.path import isfile, join
import rasterio
import tifffile as tiff

potsdam = './potsdam/2_Ortho_RGB/'
new_potsdam = './potsdam/Images/'
files = [f for f in listdir(potsdam) if isfile(join(potsdam, f))]

for f in files:
    img = rasterio.open(potsdam+f).read().transpose([1,2,0])
    tiff.imsave(new_potsdam+f, img)
