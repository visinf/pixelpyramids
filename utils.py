import os
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from data.celeba import CelebA as custom_celeba
from data.lsun import LSUN as custom_lsun
from data.celeba_1024 import CelebA as custom_celeba1024

def get_dataset( dataset_name, batch_size, data_root=None, num_levels=13, n_bits=5, train_workers=4, test_workers=2 ):

	if dataset_name == 'celeba_256':
		if data_root is None:
			data_root = '../celeba_data/celeba_data/'

		image_shape = [256,256,3]

		trainset = custom_celeba(root=os.path.join(data_root,'train'), split="train", num_levels=num_levels, n_bits=n_bits, 
			_mod=True, transform=None)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=4)

		testset = custom_celeba(root=os.path.join(data_root,'validation'), split="validation", num_levels=num_levels, n_bits=n_bits, 
			_mod=True, transform=None)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=4)

	elif dataset_name == 'celeba_1024':
		if data_root is None:
			data_root = '../celeba-hq/celeba_data/'

		image_shape = [1024,1024,3]

		transform_train = transforms.Compose([ 
			transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		transform_test = transforms.Compose([ transforms.ToTensor(),  
			transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

		trainset = custom_celeba1024(root=os.path.join(data_root,'train'), split="train", num_levels=num_levels, 
			n_bits=n_bits, patch_train=True, transform=transform_train)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=4)

		testset = custom_celeba1024(root=os.path.join(data_root,'validation'), split="validation", num_levels=num_levels,  
			n_bits=n_bits,  patch_train=False, transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=4)

	elif 'lsun' in dataset_name:

		lsun_type = dataset_name.split('_')[-1]

		if data_root is None:
			data_root = '../lsun_data/lsun'

		image_shape = [128,128,3]
		transform_train = transforms.Compose([ 
			transforms.CenterCrop(256),
			transforms.Resize(128)])

		transform_test = transforms.Compose([ 
			transforms.CenterCrop(256),
			transforms.Resize(128)])

		trainset = custom_lsun(root=data_root, classes=[lsun_type + '_train'], num_levels=num_levels, n_bits=n_bits, 
			_mod=False, transform=transform_train)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=4)

		testset = custom_lsun(root=data_root, classes=[lsun_type + '_val'], num_levels=num_levels, n_bits=n_bits, 
			_mod=False, transform=transform_test)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=4)


	return train_loader, test_loader, image_shape


