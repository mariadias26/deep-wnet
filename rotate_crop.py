import os
from PIL import Image
import tifffile as tiff
import math
import numpy as np
import cv2

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def get_map(color_codes):
  color_map = np.ndarray(shape=(256*256*256), dtype='int32')
  color_map[:] = -1
  for rgb, idx in color_codes.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx
  return color_map

def norm_image(image):
    color_codes = {
      (255, 255, 255): 1,   #imp surface
      (255, 255, 0): 2,     #car
      (0, 0, 255): 3,       #building
      (255, 0, 0): 4,       #background
      (0, 255, 255): 5,     #low veg
      (0, 255, 0): 6        #tree
    }

    color_map = get_map(color_codes)

    image = image.astype(int)
    image = image.dot(np.array([65536, 256, 1], dtype='int32'))

    new_a = color_map[image]
    image_norm = (np.arange(new_a.max()) == new_a[...,None]-1).astype(int)
    return image_norm

dataset = input('Potsdam (p) or Vaihingen (v) dataset? ')
while True:
    if dataset == 'p':
        path_img = '/home/mdias/datasets/potsdam/Images_l/'
        new_path_img = '/home/mdias/datasets/potsdam/Images_l_rot_crop/'
        path_mask = '/home/mdias/datasets/potsdam/Masks/'
        new_path_mask = '/home/mdias/datasets/potsdam/Masks_rot_crop/'
        break
    elif dataset == 'v':
        path_full_img = '/home/mdias/datasets/vaihingen/Images_lab_hist/'
        path_img = '/home/mdias/datasets/vaihingen/Images_l/'
        path_mask = '/home/mdias/datasets/vaihingen/Ground_Truth/'
        new_path_mask = '/home/mdias/datasets/vaihingen/Masks/'
        break
    else:
        dataset = input('p or v?')


train_ids = ['1', '3', '11', '13', '15', '17', '21', '26', '28', '30', '32', '34']
name_template = 'top_mosaic_09cm_area{}.tif'
new_name_template = 'top_mosaic_09cm_area_rc_{}.tif'

for f in train_ids:
    full_image = tiff.imread(path_full_img + name_template.format(f))
    image = tiff.imread(path_img + name_template.format(f))
    mask = tiff.imread(path_mask + name_template.format(f))
    image_height, image_width = image.shape[0:2]

    image_orig = np.copy(image)
    mask_orig = np.copy(mask)
    full_image_orig = np.copy(full_image)

    image_rotated = rotate_image(image, 45)
    mask_rotated = rotate_image(mask, 45)
    full_image_rotated = rotate_image(full_image, 45)

    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(45)
        )
    )

    mask_rotated_cropped = crop_around_center(
        mask_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(45)
        )
    )

    full_image_rotated_cropped = crop_around_center(
        full_image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(45)
        )
    )

    height, width = image_rotated_cropped.shape[0:2]
    image_rotated_cropped = image_rotated_cropped.reshape([height, width, 1])

    height, width = mask_rotated_cropped.shape[0:2]
    mask_rotated_cropped = norm_image(mask_rotated_cropped).astype('uint8')

    tiff.imsave(path_img + new_name_template.format(f), image_rotated_cropped)
    tiff.imsave(new_path_mask + new_name_template.format(f), mask_rotated_cropped)
    tiff.imsave(path_full_img + new_name_template.format(f), full_image_rotated_cropped)
