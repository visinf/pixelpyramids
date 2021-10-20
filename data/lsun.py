from torchvision.datasets import VisionDataset
from PIL import Image
import torch
import os
import os.path
import io
import sys
import string
from collections.abc import Iterable
import pickle
import numpy as np
from scipy import ndimage
from typing import Any, Callable, cast, List, Optional, Tuple, Union
from torchvision.datasets.utils import verify_str_arg, iterable_to_str


class LSUNClass(VisionDataset):
    def __init__(
            self, root: str, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        import lmdb
        super(LSUNClass, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length


class LSUN(VisionDataset):
    """
    `LSUN <https://www.yf.io/p/lsun>`_ dataset.

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
            self,
            root: str,
            classes: Union[str, List[str]] = "train",
            num_levels: int = 5,
            filter_size: int = 2,
            n_bits:int = 8,
            _mod: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)
        self.num_levels = num_levels
        self.filter_size = filter_size
        self._mod = _mod

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=root + '/' + c + '_lmdb',
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count
        self.n_bits = n_bits
        print('self.length ',self.length)

    def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']

        try:
            classes = cast(str, classes)
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr_type = ("Expected type str for elements in argument classes, "
                               "but got type {}.")
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes
    '''
    def downsampled_img(self,im):
        kernel = np.ones((self.filter_size,self.filter_size))*(1/(self.filter_size*self.filter_size))
        image_1 = ndimage.convolve(im[:,:,0], kernel, mode='mirror').astype(np.int64)
        image_2 = ndimage.convolve(im[:,:,1], kernel, mode='mirror').astype(np.int64)
        image_3 = ndimage.convolve(im[:,:,2], kernel, mode='mirror').astype(np.int64)
        #print(image_1.dtype,image_2.dtype,image_3.dtype)
        image = np.concatenate((np.expand_dims(image_1,axis=2),np.expand_dims(image_2,axis=2),np.expand_dims(image_3,axis=2)),axis=2)
        image = image[0::2,0::2,:]
        #image = (np.floor(image*255).astype(np.float64))/255#np.floor.astype(np.float32)#
        return image
        #images.append(np.expand_dims(image,axis=0))
        #images = np.concatenate(images,axis=0).astype(np.uint8)
        #return torch.tensor(images, dtype=torch.float32).cuda()

    def upsampled_img(self,input):
        #with torch.no_grad():
        return input.repeat(2, axis=0).repeat(2, axis=1)#torch.tensor(input.repeat(2, axis=1).repeat(2, axis=2),dtype=torch.float32)#
    '''
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


            diff_imgs.append(diff_img.permute(2,0,1))
            downsampled_imgs.append(downsmp_img.permute(2,0,1))

            #diff_imgs.append(((diff_img/(n_bins - 1) - _factor)*2).permute(2,0,1))
            #downsampled_imgs.append(((downsmp_img/(n_bins - 1)-_factor)*2.).permute(2,0,1))#/n_bins
            
            
        return downsampled_imgs, diff_imgs


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        img, _ = db[index]
        #sys.exit(0)
        img = np.array(img)
        #print('img ',img.shape)
        #img = np.transpose(img,(1,2,0))
        #print('img ',img.shape, np.amax(img),np.amin(img))
        #img = np.transpose(img,(1,2,0))

        n_bins = 2. ** self.n_bits
        if self.n_bits < 8:
            img = np.floor(img.astype(np.uint8) / 2 ** (8 - self.n_bits))

        downsampled_imgs, diff_imgs = self.get_image_pyramid(img, n_bins)

        img = img / (2**self.n_bits)
        img = 2*img - 1.
        img = np.transpose(img,(2,0,1))

        return {'diff_im':diff_imgs ,'downsmp_im':downsampled_imgs,'org_img':img}

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return "Classes: {classes}".format(**self.__dict__)