import tifffile as tiff

def get_step(test_id):
    #test = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
    dir_img = './vaihingen/test/Images_lab/top_mosaic_09cm_area{}.tif'
    patch_sz = 320
    path_img = dir_img.format(test_id)
    img = tiff.imread(path_img)
    x_original, y_original, ch = img.shape
    x, y = x_original, y_original
    step_x = []
    step_y = []
    while not step_x:
        for i in range(14, 40):
            if (x-patch_sz)%i == 0:
                step_x.append(i)
                break
        if not step_x:
            x+=1
    while not step_y:
        for i in range(14, 40):
            if (y-patch_sz)%i == 0:
                step_y.append(i)
                break
        if not step_y:
            y+=1
    return x - x_original, y - y_original, step_x[0], step_y[0]

test = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
def get_step(test_id):
    dir_img = './vaihingen/test/Images_lab/top_mosaic_09cm_area{}.tif'
    patch_sz = 320
    path_img = dir_img.format(test_id)
    img = tiff.imread(path_img)
    x_original, y_original, ch = img.shape
    x, y = x_original, y_original
    step = []
    oi = True
    while not step:
        for i in range(10, 80):
            if (x-patch_sz)%i == 0 and (y-patch_sz)%i == 0:
                step.append(i)
                break

        if not step:
            if oi:
                x+=1
                oi = False
            else:
                y+=1
                oi = True
        print(x - x_original, y - y_original)

    return x - x_original, y - y_original, step[0]

for test_id in test:
    print(get_step(test_id))
