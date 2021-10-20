# PixelPyramids: Exact Inference Models from Lossless Image Pyramids

<p align="center">
  <img width="320" height="150" src="/assets/celeba_256.png" hspace="30">
  <img width="320" height="150" src="/assets/celeba1024.png" hspace="30">
</p>

This repository is the PyTorch implementation of the paper:

[**PixelPyramids: Exact Inference Models from Lossless Image Pyramids (ICCV 2021)**](https://openaccess.thecvf.com/content/ICCV2021/html/Mahajan_PixelPyramids_Exact_Inference_Models_From_Lossless_Image_Pyramids_ICCV_2021_paper.html)

[Shweta Mahajan](https://www.visinf.tu-darmstadt.de/visinf/team_members/smahajan/smahajan.en.jsp) and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp)


## Requirements
The following code is written in Python 3.6.10 and CUDA 9.0.

Requirements:
- torch 1.7.1
- torchvision 0.8.2
- tqdm 4.58.0
- numpy 1.19.1


To install requirements:

```setup
conda config --add channels pytorch
conda config --add channels anaconda
conda config --add channels conda-forge
conda config --add channels conda-forge/label/cf202003
conda create -n <environment_name> --file requirements.txt
conda activate <environment_name>
```

## Datasets

The datasets used in this project are:
- [CelebA-HQ 256](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
- [CelebA-HQ 1024](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [LSUN-bedroom 128](https://github.com/fyu/lsun)
- [LSUN-church outdoor 128](https://github.com/fyu/lsun)
- [LSUN-tower 128](https://github.com/fyu/lsun)
- [ImageNet 128](http://www.image-net.org/download)



## Training
The important keyword arguments for training are,
- params_file : Path to the configuration file in the params folder.
- dataset_name : Name of the dataset. This can be found in the config files (params folder) of the different datasets.
- data_root : Path to the location of the dataset. Please see `utils.py` for default values.
- L : Number of levels in the pyramid decomposition.
- C : Number of channels for output of U-Net.
- n_bits : Number of bits for training the data. 
- n_classes : List of the (number of mixture components) x 10. For each mixture component, 3 params for means, 3 params for 	coefficients, 3 params for logscales, 1 param for logits
- n_squeeze : List of number of squeeze operations per level.
- n_channels : Number of channels for the PixelCNNPP at the coarsest level.
- n_res_layers : Number of residual layers for the PixelCNNPP at the coarsest level.

Please follow the following instructions for training:
1. Train a model on CelebA-HQ-256,
 ```
		python main.py --params_file './params/celeba_256.json' 
 ```
2. The model is evaluated after every epoch

## Generation and Validation

Samples and test results in bits/dim can be obtained using `main.py`. Generated samples are stored in the `./samples` folder. Download the [checkpoints](https://drive.google.com/drive/folders/1F74VFrmW8P6WMUZ7pLqvtVX0ZMY6HMK-?usp=sharing) to the `ckpts` folder.

### Memory requirements
The models were trained on four nvidia V100 GPU with 32 GB memory. The levels can be trained in parallel with a maximum of 24GB memory per level.


## Results


### Evaluation on PixelPramids 5-bit 

|              	   |    bits/dim    |  
| ---------------- | -------------- |
| CelebA-HQ_256	   |      0.61      | 
| CelebA-HQ_1024   |      0.58      |
| LSUN_bedroom_128 |      0.88      |
| LSUN_church_128  |      1.07      |
| LSUN_tower_128   |      0.95      |
| ImageNet_128     |      3.40      | 

## Bibtex

	@inproceedings{pixelpyramids21iccv,
	  title     = {PixelPyramids: Exact Inference Models from Lossless Image Pyramids},
	  author    = {Mahajan, Shweta and Roth, Stefan},
	  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	  year = {2021}
	}
