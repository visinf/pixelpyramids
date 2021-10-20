from functools import partial
import torch
import os
from PIL import Image
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets import VisionDataset
import numpy as np
import sys
import random
from scipy import ndimage

class CelebA(VisionDataset):
    
    def __init__(
            self,
            root: str,
            split: str = "train",
            num_levels: int = 5,
            filter_size: int = 2,
            n_bits: int = 8,
            patch_train: bool = True,
            transform: Optional[Callable] = None,
    ) -> None:
        super(CelebA, self).__init__(root, transform=transform)
        self.split = split
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.n_bits = n_bits
        self.patch_train = patch_train
        
        target_dir = os.path.join(self.root, self.split)
        instances = []
        print('target_dir ',target_dir)
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                #print(split, fname)
                instances.append(path)
        self.samples = instances
    
    
    def get_image_pair(self, input, n_bins):
        if input.shape[0] == input.shape[1]:
            image_p1 = input[0::2,:,:] 
            image_p2 = input[1::2,:,:] 
            diff_img = np.mod(image_p1.astype(np.int64) - image_p2.astype(np.int64), n_bins) # image_p1 = torch.remainder(diff_img + image_p2, n_bins)
            downsmp_img = image_p2
            return downsmp_img.astype(np.uint8), diff_img.astype(np.uint8)
        else:
            image_p1 = input[:,0::2,:] 
            image_p2 = input[:,1::2,:] 
            diff_img = np.mod(image_p1.astype(np.int64) - image_p2.astype(np.int64), n_bins)
            downsmp_img = image_p2
            return downsmp_img.astype(np.uint8), diff_img.astype(np.uint8)


    def get_image_pyramid(self,input, n_bins):
        downsampled_imgs = []
        diff_imgs = []
        _factor = ((n_bins)/2.)/(n_bins - 1)
        for l in range(self.num_levels-1):
            downsmp_img, diff_img = self.get_image_pair(input, n_bins)
            input = downsmp_img.copy()

            downsmp_img = torch.tensor(downsmp_img,dtype=torch.float32)
            diff_img = torch.tensor(diff_img,dtype=torch.float32)

            downsmp_img = downsmp_img / (n_bins - 1)
            diff_img = diff_img / (n_bins - 1)

            downsmp_img = 2.*downsmp_img
            diff_img = 2.*diff_img

            downsmp_img = downsmp_img - 1.#127.5/(n_bins - 1.)
            diff_img = diff_img - 1.#127.5/(n_bins - 1.)

            diff_img = diff_img.permute(2,0,1)
            downsmp_img = downsmp_img.permute(2,0,1)

            if self.patch_train and l < 3:
                start_idx_x = random.choice(list(range(0, diff_img.size(1) - 256 + 1)))
                start_idx_y = random.choice(list(range(0, diff_img.size(2) - 256 + 1)))
                diff_img = diff_img[:,start_idx_x:start_idx_x+256,start_idx_y:start_idx_y+256]
                downsmp_img = downsmp_img[:,start_idx_x:start_idx_x+256,start_idx_y:start_idx_y+256]

            diff_imgs.append(diff_img)
            downsampled_imgs.append(downsmp_img)

            #diff_imgs.append(((diff_img/(n_bins - 1) - _factor)*2).permute(2,0,1))
            #downsampled_imgs.append(((downsmp_img/(n_bins - 1)-_factor)*2.).permute(2,0,1))#/n_bins
            
            
        return downsampled_imgs, diff_imgs


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x = np.array(Image.open(self.samples[index]))
        n_bins = 2. ** self.n_bits
        if self.n_bits < 8:
            x = np.floor(x / 2 ** (8 - self.n_bits))
        downsampled_imgs, diff_imgs = self.get_image_pyramid(x, n_bins)
        sr_low_res = x.copy()
        sr_low_res = Image.fromarray(sr_low_res.astype(np.uint8))
        sr_low_res = sr_low_res.resize((16,16), Image.BICUBIC)
        sr_low_res = np.array(sr_low_res)

        x = x / (2**self.n_bits)
        x = 2*x - 1.

        sr_low_res = sr_low_res / (2**self.n_bits)
        sr_low_res =2*sr_low_res - 1.

        x = np.transpose(x,(2,0,1))
        sr_low_res = np.transpose(sr_low_res,(2,0,1))
        #print('x ',x.shape) , 'input_im':np.transpose(x,(2,0,1))
        return {'diff_im':diff_imgs ,'downsmp_im':downsampled_imgs,
        'org_im':torch.tensor(x).float(),'sr_low_res':sr_low_res}

    def __len__(self) -> int:
        return len(self.samples)
