import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
import cv2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def resize_(image, label, output_size):
    w = output_size[0]
    h = output_size[1]

    iw, ih= image.size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image       = image.resize((nw,nh), Image.BICUBIC)
    new_image   = Image.new('RGB', [w, h], (0, 0, 0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label       = label.resize((nw,nh), Image.NEAREST)
    new_label   = Image.new('L', [w, h], (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    
    return new_image, new_label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = resize_(image, label, self.output_size)

        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        # x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = np.array(image)
        label = np.array(label)

        image = image.transpose((2,0,1))
        # label = label.transpose((2,1,0))

        image = torch.from_numpy(image.astype(np.float32))       
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            image = Image.open(os.path.join(self.data_dir, "JPEGImages", slice_name+'.jpg')).convert('RGB')
            label = Image.open(os.path.join(self.data_dir, "SegmentationClass", slice_name+'.png'))
            label = np.array(label)
            label[label > 0] = 1     #这里我把255转到了1
            label = Image.fromarray(np.uint8(label))
            # image = image.transpose(2,1,0)
            # label = label.transpose(2,1,0)
        else:
            slice_name = self.sample_list[idx].strip('\n')
            image = Image.open(os.path.join(self.data_dir, "JPEGImages", slice_name+'.jpg')).convert('RGB')
            label = Image.open(os.path.join(self.data_dir, "SegmentationClass", slice_name+'.png'))
            label = np.array(label)
            label[label > 0] = 1     #这里我把255转到了1
            label = Image.fromarray(np.uint8(label))

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
