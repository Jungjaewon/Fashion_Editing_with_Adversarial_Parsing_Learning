import glob
import os
import os.path as osp
import cv2
import numpy as np
import pickle

from tqdm import tqdm

color_rgb = np.array(
            [(0, 0, 0),# 0=background
             (198, 226, 244),# 1=beige
             (50, 50, 50), # 2=black
             (255, 32, 0), # 3=blue
             (77, 96, 128), #4=brown
             (128, 128, 128), # gray
             (110, 198, 119), # green
             (0, 131, 255), # orange
             (193, 182, 255), # pink
             (169, 81, 120), # purple
             (36, 28, 237), # red
             (239, 243, 243), # white
             (0, 255, 255), # yellow
             ])


def decode_segmap(image):
    label_colors = color_rgb
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, len(color_rgb)):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=0)
    rgb = np.squeeze(rgb)
    return np.transpose(rgb, (1, 2, 0))

def do_rename():

    for img_path in tqdm(glob.glob(osp.join('image', '*.jpg'))):
        img_name = img_path.split(os.sep)[-1].split('.')[0]
        dir_name = osp.dirname(img_path)
        new_path = osp.join(dir_name, f'{img_name}_image.jpg')
        os.rename(img_path, new_path)


def get_canny():
    for img_path in tqdm(glob.glob(osp.join('image', '*.jpg'))):
        img_name = img_path.split(os.sep)[-1].split('.')[0].split('_')[0]
        dir_name = osp.dirname(img_path)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_canny = cv2.Canny(img_gray, 50, 150)
        cv2.imwrite(osp.join(dir_name,f'{img_name}_sketch.jpg'), img_canny)


def load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def load_color_mask():

    data_list = load_pickle('label/color_segment_train.plk')
    data_list.extend(load_pickle('label/color_segment_test.plk'))

    for data in tqdm(data_list):
        mask = data['semseg']
        mask = mask.astype(np.uint8)
        mask_rgb = decode_segmap(mask)
        image_name = data['img_name'].split('.')[0]
        cv2.imwrite(osp.join('image', f'{image_name}_color.jpg'), mask_rgb)


def make_bin():
    for img_path in tqdm(glob.glob(osp.join('image', '*_color.jpg'))):
        img_name = img_path.split(os.sep)[-1].split('.')[0].split('_')[0]
        dir_name = osp.dirname(img_path)
        img_color = cv2.imread(img_path)
        bin = np.zeros_like(img_color)
        for y in range(len(bin)):
            for x in range(len(bin[y])):
                if not np.array_equal(img_color[y][x], np.array([0,0,0])):
                    bin[y][x] = np.array([255, 255, 255])

        cv2.imwrite(osp.join(dir_name, f'{img_name}_bin.jpg'), bin)


def cut_background():
    for img_path in tqdm(glob.glob(osp.join('image', '*_sketch.jpg'))):
        img_name = img_path.split(os.sep)[-1].split('.')[0].split('_')[0]
        dir_name = osp.dirname(img_path)

        sketch = cv2.imread(img_path)
        color = cv2.imread(osp.join(dir_name, f'{img_name}_color.jpg'))
        image = cv2.imread(osp.join(dir_name, f'{img_name}_image.jpg'))

        for y in range(len(color)):
            for x in range(len(color[y])):
                if np.array_equal(color[y][x], np.array([0,0,0])):
                    sketch[y][x] = np.array([0, 0, 0])
                    image[y][x] = np.array([0, 0, 0])
        cv2.imwrite(osp.join(dir_name, f'{img_name}_sketch.jpg'), sketch)
        cv2.imwrite(osp.join(dir_name, f'{img_name}_image.jpg'), image)

if __name__ == '__main__':
    pass
    #do_rename()
    #get_canny()
    #load_color_mask()
    #make_bin()
    cut_background()