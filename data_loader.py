import os
import os.path as osp
import glob
import pickle
import torch
import numpy as np
import random
import cv2
import torchvision.transforms.functional as TF

from torch.utils import data
from torchvision import transforms as T
from PIL import Image


def load_pickle(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def random_ff_mask(H, W, max_angle=4, max_len=40, max_width=10, times=10):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = H
        width = W
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1,) + mask.shape).astype(np.float32)


class InpaintDataSet(data.Dataset):

    def __init__(self, config, transform, mode):

        assert mode in ['train', 'test']
        self.transform = transform
        self.mode = mode
        self.img_dir = config['TRAINING_CONFIG']['INPAINT_IMG_DIR'] # , config['TRAINING_CONFIG']['MODE']
        self.H, self.W = config['MODEL_CONFIG']['IMG_SIZE'].split(",")
        self.H, self.W = int(self.H), int(self.W)
        self.data_list = glob.glob(osp.join(self.img_dir, '*'))
        self.data_list = [x.split(os.sep)[-1].split('_')[0] for x in self.data_list]
        print(f'num of data : {len(self.data_list)}')

    def transform_func(self, image, bin, sketch, color, seg_map):
        # Resize
        resize = T.Resize((self.H, self.W))
        image = self.transform(resize(image))
        bin = self.transform(resize(bin))
        sketch = self.transform(resize(sketch))
        color = self.transform(resize(color))
        seg_map = self.transform(resize(seg_map))

        # Random horizontal flipping
        if self.mode == 'train':
            if random.random() > 0.5:
                image, bin, sketch, color, seg_map = TF.hflip(image), TF.hflip(bin), TF.hflip(sketch), TF.hflip(color), TF.hflip(seg_map)

            # Random vertical flipping
            if random.random() > 0.5:
                image, bin, sketch, color, seg_map = TF.vflip(image), TF.vflip(bin), TF.vflip(sketch), TF.vflip(color), TF.vflip(seg_map)

        return image, bin, sketch, color, seg_map

    def __getitem__(self, index):
        id = self.data_list[index]

        img_path = osp.join(self.img_dir, f'{id}_image.jpg')
        bin_path = osp.join(self.img_dir, f'{id}_bin.jpg')
        sketch_path = osp.join(self.img_dir, f'{id}_sketch.jpg')
        color_path = osp.join(self.img_dir, f'{id}_color.jpg')
        segmap_path = osp.join(self.img_dir, f'{id}_segmap.plk')

        seg_map = load_pickle(segmap_path)
        seg_map = torch.from_numpy(seg_map.astype(np.uint8)).long()

        image = Image.open(img_path).convert('RGB')
        bin = Image.open(bin_path).convert('L')
        sketch = Image.open(sketch_path).convert('L')
        color = Image.open(color_path).convert('RGB')
        image, bin, sketch, color, seg_map = self.transform_func(image, bin, sketch, color, seg_map)

        mask = torch.from_numpy(random_ff_mask(self.H, self.W)).contiguous()

        in_image = image * (1 - mask)
        inseg_map = seg_map * (1 - mask)
        bin = bin * (1 - mask)
        sketch = sketch * mask
        color = color * mask
        noise = torch.randn(1, self.H, self.W) * mask

        return torch.LongTensor(int(id)), image, in_image, seg_map, inseg_map, bin, sketch, color, noise, mask

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


class SegDataSet(data.Dataset):

    def __init__(self, config, transform, mode):

        assert mode in ['train', 'test']
        self.transform = transform
        self.mode = mode
        self.img_dir = config['TRAINING_CONFIG']['SEG_IMG_DIR'] # , config['TRAINING_CONFIG']['MODE']
        self.data_list = glob.glob(osp.join(self.img_dir, '*'))
        self.data_list = [x.split(os.sep)[-1].split('_')[0] for x in self.data_list]
        self.H, self.W = config['MODEL_CONFIG']['IMG_SIZE'].split(",")
        self.H, self.W = int(self.H), int(self.W)

    def transform_func(self, sketch, color, seg_map):
        # Resize
        resize = T.Resize((self.H, self.W))
        sketch = self.transform(resize(sketch))
        color = self.transform(resize(color))
        seg_map = resize(seg_map.unsqueeze(0)).squeeze()

        # Random horizontal flipping
        if self.mode == 'train':
            if random.random() > 0.5:
                sketch, color, seg_map = TF.hflip(sketch), TF.hflip(color), TF.hflip(seg_map)

            # Random vertical flipping
            if random.random() > 0.5:
                sketch, color, seg_map = TF.vflip(sketch), TF.vflip(color), TF.vflip(seg_map)

        return sketch, color, seg_map

    def __getitem__(self, index):
        id = self.data_list[index]

        sketch_path = osp.join(self.img_dir, f'{id}_sketch.jpg')
        color_path = osp.join(self.img_dir, f'{id}_color.jpg')
        segmap_path = osp.join(self.img_dir, f'{id}_segmap.plk')

        seg_map = load_pickle(segmap_path)
        seg_map = torch.from_numpy(seg_map.astype(np.uint8)).long()

        sketch = Image.open(sketch_path).convert('L')
        color = Image.open(color_path).convert('RGB')
        image, color, seg_map = self.transform_func(sketch, color, seg_map)

        mask = torch.from_numpy(random_ff_mask(self.H, self.W)).contiguous()

        inseg_map = seg_map * (1 - mask)
        sketch = sketch * mask
        color = color * mask
        noise = torch.randn(1, self.H, self.W) * mask

        return torch.LongTensor(int(id)), seg_map, inseg_map, sketch, color, noise, mask

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config, type, mode):

    img_transform = list()

    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    if mode == 'train':
        batch_size = config['TRAINING_CONFIG']['BATCH_SIZE']
    else:
        batch_size = 1

    #if type == 'seg':
    #    dataset = SegDataSet(config, img_transform, mode)
    #else:
    dataset = InpaintDataSet(config, img_transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader