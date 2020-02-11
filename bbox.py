from collections import defaultdict
import csv
import sys
import os
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def map_labels(old_label):
    if old_label == '1' or old_label == '2':
        return 0
    elif old_label == '3' or old_label == '4':
        return 1
    elif old_label == '5':
        return 2
    elif old_label == '6':
        return 3
    elif old_label == '7' or old_label == '8':
        return 4
    elif old_label == '9' or old_label == '10':
        return 5
    else:
        return -1


csv.field_size_limit(sys.maxsize);


def get_scalers(img_size, x_m, y_m):
    h, w = img_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_m, h_ / y_m


def mask_for_polygons(polygons, img_size):
    img_mask = np.zeros(img_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


path = '/home/mdias/datasets/dstl-satellite-imagery-feature-detection/'
image_ids = list(listdir_nohidden(path + 'train_geojson_v3/'))
poly_types = list(range(1, 11))
poly_types = [str(i) for i in poly_types]

for IM_ID in image_ids:
    # Read image with tiff
    im_rgb = tiff.imread(path + 'three_band/{}.tif'.format(IM_ID)).transpose([1,2,0])
    im_size = im_rgb.shape[:2]
    mask = np.zeros(shape = (7, im_size[0], im_size[1]))
    for POLY_TYPE in poly_types:
        # Load grid size
        x_max = y_min = None
        for _im_id, _x, _y in csv.reader(open(path + 'grid_sizes.csv')):
            if _im_id == IM_ID:
                x_max, y_min = float(_x), float(_y)
                break

        # Load train poly with shapely
        train_polygons = None
        for _im_id, _poly_type, _poly in csv.reader(open(path + 'train_wkt_v4.csv')):
            if _im_id == IM_ID and _poly_type == POLY_TYPE:
                train_polygons = shapely.wkt.loads(_poly)
                break

        x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
        train_polygons_scaled = shapely.affinity.scale(train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
        train_mask = mask_for_polygons(train_polygons_scaled, im_size)
        mask[map_labels(POLY_TYPE)] += train_mask
    mask = np.where(mask >= 1, 1, 0)
    mask_count = np.sum(mask, axis=0)
    oi = np.where(mask_count == 0, 1, 0)
    mask[6] = oi
    mask_count = np.sum(mask, axis=0)
    mask = mask / mask_count

    print(IM_ID)
    mask = mask.transpose([1,2,0])
    print(mask.shape)
    tiff.imsave(path + 'mask_uni/{}.tif'.format(IM_ID), mask)

