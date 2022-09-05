import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import cv2
from PIL import Image

def resize_(image, output_size):
    w = output_size[0]
    h = output_size[1]

    iw, ih= image.size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image       = image.resize((nw,nh), Image.BICUBIC)
    new_image   = Image.new('RGB', [w, h], (0, 0, 0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    
    return new_image,nw,nh

config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]

config_vit.n_classes = 2
config_vit.n_skip = 3
config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    
model = ViT_seg(config_vit, img_size=512, num_classes=2).cuda()
# print(model)
model.load_state_dict(torch.load("best.pth"))

model.eval()


data_path = "../code/mine/processed_data/"
clas = os.listdir(data_path)
for cla in clas: 
    imgs = os.listdir(data_path + cla)
    for img in imgs:
        with torch.no_grad():
            image = Image.open(data_path + cla + "/" + img).convert('RGB')
            
            origin_img = image.copy()
            orininal_h  = np.array(image).shape[0]
            orininal_w  = np.array(image).shape[1]

            image, nw, nh = resize_(image, [512,512]) 

            image = np.array(image)
            image = image.transpose((2,0,1))
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image.astype(np.float32))  
            
            image = image.cuda()   
            outputs = model(image)

            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            img_mask = out[int((512 - nh) // 2) : int((512 - nh) // 2 + nh), \
                        int((512 - nw) // 2) : int((512 - nw) // 2 + nw)]
            
            img_mask = cv2.resize(img_mask.astype("uint8"), (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            cv2.imwrite("../code/mine/new_mask/" + cla + "/" + img, img_mask)
            
            mask = [img_mask, img_mask, img_mask]
            mask = np.transpose(mask, [1,2,0])
            cv2.imwrite("../code/mine/new_res/" + cla + "/" + img, mask * origin_img)
            print(img)

#-----------------------------------------------------------------------------------------------------------------------
# with torch.no_grad():
#     image = Image.open("dengzhou_0.jpg").convert('RGB')
#     image_array = resize_(image, [512,512]) 

#     image_array = np.array(image_array)
#     image_array = image_array.transpose((2,0,1))
#     image_array = np.expand_dims(image_array, 0)
#     image_array = torch.from_numpy(image_array.astype(np.float32))  

#     image_array = image_array.cuda()   
#     outputs = model(image_array)

#     out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#     out = out.cpu().detach().numpy()

#     label = Image.fromarray(out.astype(np.uint8), mode='L')
#     label.save("dengzhou.png")


# i = Image.open("dengzhou_0.jpg").convert('RGB')
# orininal_h  = np.array(i).shape[0]
# orininal_w  = np.array(i).shape[1]

# image_array, nw,nh = resize_(i, [512,512]) 

# image = cv2.imread("dengzhou_0.jpg")
# img_mask = cv2.imread("dengzhou.png", 0)
# # mask = np.transpose(np.array([img_mask, img_mask, img_mask]), (1, 2, 0))

# img_mask = img_mask[int((512 - nh) // 2) : int((512 - nh) // 2 + nh), \
#         int((512 - nw) // 2) : int((512 - nw) // 2 + nw)]

# img_mask = cv2.resize(img_mask, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

# mask = [img_mask, img_mask, img_mask]
# mask = np.transpose(mask, [1,2,0])

# cv2.imwrite("res.jpg", mask * image)